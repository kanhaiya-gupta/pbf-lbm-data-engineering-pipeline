"""
Historical Analysis Service

This module implements the historical analysis service for PBF-LB/M processes.
It provides REST API endpoints for historical data analysis, trend analysis,
and performance evaluation over time.
"""

import logging
import time
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import mlflow
import mlflow.tensorflow
from pathlib import Path
import json
import pickle
from sqlalchemy import create_engine, text
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

from ...pipelines.inference.batch_inference import BatchInferencePipeline
from ...models.base_model import BaseModel
from ...config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


# Pydantic models for API requests and responses
class HistoricalAnalysisRequest(BaseModel):
    """Request model for historical analysis."""
    analysis_type: str = Field(..., description="Type of analysis (trend, performance, quality, defects)")
    start_date: str = Field(..., description="Start date for analysis (ISO format)")
    end_date: str = Field(..., description="End date for analysis (ISO format)")
    data_sources: List[str] = Field(..., description="List of data sources to analyze")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")
    aggregation_period: str = Field("daily", description="Aggregation period (hourly, daily, weekly, monthly)")
    metrics: List[str] = Field(..., description="List of metrics to analyze")
    output_format: str = Field("json", description="Output format (json, csv, excel)")


class HistoricalAnalysisResponse(BaseModel):
    """Response model for historical analysis."""
    analysis_id: str = Field(..., description="Analysis ID")
    analysis_type: str = Field(..., description="Type of analysis")
    start_date: str = Field(..., description="Start date")
    end_date: str = Field(..., description="End date")
    data_sources: List[str] = Field(..., description="Data sources analyzed")
    total_records: int = Field(..., description="Total records analyzed")
    analysis_results: Dict[str, Any] = Field(..., description="Analysis results")
    visualizations: Optional[Dict[str, Any]] = Field(None, description="Generated visualizations")
    output_path: Optional[str] = Field(None, description="Output file path")
    created_at: str = Field(..., description="Analysis creation timestamp")


class TrendAnalysisRequest(BaseModel):
    """Request model for trend analysis."""
    start_date: str = Field(..., description="Start date for trend analysis")
    end_date: str = Field(..., description="End date for trend analysis")
    metrics: List[str] = Field(..., description="Metrics to analyze trends for")
    data_sources: List[str] = Field(..., description="Data sources for trend analysis")
    trend_detection_method: str = Field("linear", description="Trend detection method")
    confidence_level: float = Field(0.95, description="Confidence level for trend analysis")


class PerformanceAnalysisRequest(BaseModel):
    """Request model for performance analysis."""
    start_date: str = Field(..., description="Start date for performance analysis")
    end_date: str = Field(..., description="End date for performance analysis")
    performance_metrics: List[str] = Field(..., description="Performance metrics to analyze")
    data_sources: List[str] = Field(..., description="Data sources for performance analysis")
    benchmark_values: Optional[Dict[str, float]] = Field(None, description="Benchmark values for comparison")
    target_values: Optional[Dict[str, float]] = Field(None, description="Target values for comparison")


class QualityAnalysisRequest(BaseModel):
    """Request model for quality analysis."""
    start_date: str = Field(..., description="Start date for quality analysis")
    end_date: str = Field(..., description="End date for quality analysis")
    quality_metrics: List[str] = Field(..., description="Quality metrics to analyze")
    data_sources: List[str] = Field(..., description="Data sources for quality analysis")
    quality_thresholds: Optional[Dict[str, float]] = Field(None, description="Quality thresholds")
    defect_categories: Optional[List[str]] = Field(None, description="Defect categories to analyze")


class HistoricalAnalysisService:
    """
    Historical analysis service for PBF-LB/M processes.
    
    This service provides historical analysis capabilities for:
    - Trend analysis
    - Performance evaluation
    - Quality assessment
    - Defect analysis
    - Process optimization insights
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the historical analysis service.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.app = FastAPI(
            title="Historical Analysis Service",
            description="Historical analysis for PBF-LB/M manufacturing",
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
        
        # Initialize batch inference pipeline
        self.batch_pipeline = BatchInferencePipeline(self.config_manager)
        
        # Analysis counter
        self.analysis_counter = 0
        
        # Service metrics
        self.service_metrics = {
            'total_analyses': 0,
            'trend_analyses': 0,
            'performance_analyses': 0,
            'quality_analyses': 0,
            'defect_analyses': 0,
            'last_analysis_time': None
        }
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Initialized HistoricalAnalysisService")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": "historical_analysis",
                "timestamp": datetime.now().isoformat(),
                "metrics": self.service_metrics
            }
        
        @self.app.post("/analyze", response_model=HistoricalAnalysisResponse)
        async def perform_historical_analysis(request: HistoricalAnalysisRequest):
            """Perform historical analysis."""
            return await self._perform_historical_analysis(request)
        
        @self.app.post("/trends", response_model=HistoricalAnalysisResponse)
        async def analyze_trends(request: TrendAnalysisRequest):
            """Analyze trends in historical data."""
            return await self._analyze_trends(request)
        
        @self.app.post("/performance", response_model=HistoricalAnalysisResponse)
        async def analyze_performance(request: PerformanceAnalysisRequest):
            """Analyze performance over time."""
            return await self._analyze_performance(request)
        
        @self.app.post("/quality", response_model=HistoricalAnalysisResponse)
        async def analyze_quality(request: QualityAnalysisRequest):
            """Analyze quality trends."""
            return await self._analyze_quality(request)
        
        @self.app.get("/analyses")
        async def list_analyses(limit: int = Query(10, ge=1, le=100)):
            """List recent analyses."""
            return await self._list_analyses(limit)
        
        @self.app.get("/analyses/{analysis_id}")
        async def get_analysis(analysis_id: str):
            """Get analysis results."""
            return await self._get_analysis(analysis_id)
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get service metrics."""
            return self.service_metrics
    
    async def _perform_historical_analysis(self, request: HistoricalAnalysisRequest) -> HistoricalAnalysisResponse:
        """
        Perform historical analysis.
        
        Args:
            request: Historical analysis request
            
        Returns:
            Historical analysis response
        """
        # Generate analysis ID
        self.analysis_counter += 1
        analysis_id = f"historical_analysis_{self.analysis_counter}_{int(time.time())}"
        
        try:
            # Load historical data
            historical_data = await self._load_historical_data(
                start_date=request.start_date,
                end_date=request.end_date,
                data_sources=request.data_sources,
                filters=request.filters
            )
            
            # Perform analysis based on type
            if request.analysis_type == "trend":
                results = await self._perform_trend_analysis(historical_data, request)
            elif request.analysis_type == "performance":
                results = await self._perform_performance_analysis(historical_data, request)
            elif request.analysis_type == "quality":
                results = await self._perform_quality_analysis(historical_data, request)
            elif request.analysis_type == "defects":
                results = await self._perform_defect_analysis(historical_data, request)
            else:
                raise ValueError(f"Unsupported analysis type: {request.analysis_type}")
            
            # Generate visualizations
            visualizations = await self._generate_visualizations(historical_data, results, request)
            
            # Save results
            output_path = await self._save_analysis_results(analysis_id, results, request)
            
            # Update metrics
            self.service_metrics['total_analyses'] += 1
            self.service_metrics[f'{request.analysis_type}_analyses'] += 1
            self.service_metrics['last_analysis_time'] = datetime.now().isoformat()
            
            return HistoricalAnalysisResponse(
                analysis_id=analysis_id,
                analysis_type=request.analysis_type,
                start_date=request.start_date,
                end_date=request.end_date,
                data_sources=request.data_sources,
                total_records=len(historical_data),
                analysis_results=results,
                visualizations=visualizations,
                output_path=output_path,
                created_at=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Historical analysis failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _load_historical_data(self, start_date: str, end_date: str, 
                                  data_sources: List[str], filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Load historical data from specified sources."""
        # This would implement actual data loading logic
        # For now, return mock data
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        
        mock_data = {
            'timestamp': date_range,
            'temperature': np.random.normal(1000, 50, len(date_range)),
            'pressure': np.random.normal(1.0, 0.1, len(date_range)),
            'laser_power': np.random.normal(200, 20, len(date_range)),
            'scan_speed': np.random.normal(1000, 100, len(date_range)),
            'quality_score': np.random.uniform(0.7, 1.0, len(date_range)),
            'defect_count': np.random.poisson(2, len(date_range)),
            'build_time': np.random.normal(3600, 300, len(date_range))
        }
        
        df = pd.DataFrame(mock_data)
        
        # Apply filters if provided
        if filters:
            for key, value in filters.items():
                if key in df.columns:
                    if isinstance(value, list):
                        df = df[df[key].isin(value)]
                    else:
                        df = df[df[key] == value]
        
        return df
    
    async def _perform_trend_analysis(self, data: pd.DataFrame, request: HistoricalAnalysisRequest) -> Dict[str, Any]:
        """Perform trend analysis."""
        results = {}
        
        for metric in request.metrics:
            if metric in data.columns:
                # Calculate trend using linear regression
                x = np.arange(len(data))
                y = data[metric].values
                
                # Simple linear regression
                slope, intercept = np.polyfit(x, y, 1)
                trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
                
                # Calculate R-squared
                y_pred = slope * x + intercept
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                results[metric] = {
                    'trend_direction': trend_direction,
                    'slope': float(slope),
                    'intercept': float(intercept),
                    'r_squared': float(r_squared),
                    'trend_strength': 'strong' if r_squared > 0.7 else 'moderate' if r_squared > 0.3 else 'weak',
                    'current_value': float(y[-1]),
                    'change_percentage': float((y[-1] - y[0]) / y[0] * 100) if y[0] != 0 else 0
                }
        
        return results
    
    async def _perform_performance_analysis(self, data: pd.DataFrame, request: HistoricalAnalysisRequest) -> Dict[str, Any]:
        """Perform performance analysis."""
        results = {}
        
        for metric in request.metrics:
            if metric in data.columns:
                values = data[metric].values
                
                results[metric] = {
                    'mean': float(np.mean(values)),
                    'median': float(np.median(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'percentile_25': float(np.percentile(values, 25)),
                    'percentile_75': float(np.percentile(values, 75)),
                    'coefficient_of_variation': float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else 0,
                    'stability_score': float(1 - (np.std(values) / np.mean(values))) if np.mean(values) != 0 else 0
                }
        
        return results
    
    async def _perform_quality_analysis(self, data: pd.DataFrame, request: HistoricalAnalysisRequest) -> Dict[str, Any]:
        """Perform quality analysis."""
        results = {}
        
        # Quality score analysis
        if 'quality_score' in data.columns:
            quality_scores = data['quality_score'].values
            results['quality_score'] = {
                'mean_quality': float(np.mean(quality_scores)),
                'quality_consistency': float(1 - np.std(quality_scores)),
                'high_quality_percentage': float(np.sum(quality_scores > 0.8) / len(quality_scores) * 100),
                'low_quality_percentage': float(np.sum(quality_scores < 0.6) / len(quality_scores) * 100),
                'quality_trend': await self._calculate_trend(quality_scores)
            }
        
        # Defect analysis
        if 'defect_count' in data.columns:
            defect_counts = data['defect_count'].values
            results['defects'] = {
                'total_defects': int(np.sum(defect_counts)),
                'mean_defects_per_unit': float(np.mean(defect_counts)),
                'defect_rate': float(np.sum(defect_counts > 0) / len(defect_counts) * 100),
                'defect_trend': await self._calculate_trend(defect_counts)
            }
        
        return results
    
    async def _perform_defect_analysis(self, data: pd.DataFrame, request: HistoricalAnalysisRequest) -> Dict[str, Any]:
        """Perform defect analysis."""
        results = {}
        
        if 'defect_count' in data.columns:
            defect_counts = data['defect_count'].values
            
            results['defect_summary'] = {
                'total_defects': int(np.sum(defect_counts)),
                'mean_defects': float(np.mean(defect_counts)),
                'max_defects': int(np.max(defect_counts)),
                'defect_frequency': float(np.sum(defect_counts > 0) / len(defect_counts)),
                'zero_defect_percentage': float(np.sum(defect_counts == 0) / len(defect_counts) * 100)
            }
            
            # Defect patterns
            results['defect_patterns'] = {
                'peak_defect_periods': await self._identify_peak_periods(data, 'defect_count'),
                'defect_correlation': await self._analyze_defect_correlations(data),
                'defect_trend': await self._calculate_trend(defect_counts)
            }
        
        return results
    
    async def _calculate_trend(self, values: np.ndarray) -> Dict[str, Any]:
        """Calculate trend for a series of values."""
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        return {
            'direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
            'slope': float(slope),
            'change_rate': float(slope * len(values))
        }
    
    async def _identify_peak_periods(self, data: pd.DataFrame, column: str) -> List[Dict[str, Any]]:
        """Identify peak periods in the data."""
        values = data[column].values
        mean_val = np.mean(values)
        std_val = np.std(values)
        threshold = mean_val + 2 * std_val
        
        peak_indices = np.where(values > threshold)[0]
        peak_periods = []
        
        for idx in peak_indices:
            peak_periods.append({
                'timestamp': data.iloc[idx]['timestamp'].isoformat(),
                'value': float(values[idx]),
                'severity': 'high' if values[idx] > mean_val + 3 * std_val else 'medium'
            })
        
        return peak_periods
    
    async def _analyze_defect_correlations(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze correlations between defects and other metrics."""
        correlations = {}
        
        if 'defect_count' in data.columns:
            for column in data.columns:
                if column != 'defect_count' and column != 'timestamp':
                    try:
                        corr = data['defect_count'].corr(data[column])
                        if not np.isnan(corr):
                            correlations[column] = float(corr)
                    except Exception:
                        continue
        
        return correlations
    
    async def _generate_visualizations(self, data: pd.DataFrame, results: Dict[str, Any], 
                                     request: HistoricalAnalysisRequest) -> Dict[str, Any]:
        """Generate visualizations for the analysis."""
        visualizations = {}
        
        try:
            # Time series plot
            if 'timestamp' in data.columns:
                fig = go.Figure()
                
                for metric in request.metrics:
                    if metric in data.columns:
                        fig.add_trace(go.Scatter(
                            x=data['timestamp'],
                            y=data[metric],
                            mode='lines',
                            name=metric,
                            line=dict(width=2)
                        ))
                
                fig.update_layout(
                    title=f"{request.analysis_type.title()} Analysis - {request.start_date} to {request.end_date}",
                    xaxis_title="Time",
                    yaxis_title="Value",
                    hovermode='x unified'
                )
                
                visualizations['time_series'] = json.loads(fig.to_json())
            
            # Distribution plots
            for metric in request.metrics:
                if metric in data.columns:
                    fig = px.histogram(data, x=metric, title=f"{metric} Distribution")
                    visualizations[f'{metric}_distribution'] = json.loads(fig.to_json())
            
        except Exception as e:
            logger.warning(f"Could not generate visualizations: {e}")
        
        return visualizations
    
    async def _save_analysis_results(self, analysis_id: str, results: Dict[str, Any], 
                                   request: HistoricalAnalysisRequest) -> str:
        """Save analysis results to file."""
        output_dir = Path("outputs/historical_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if request.output_format == "json":
            output_path = output_dir / f"{analysis_id}.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        elif request.output_format == "csv":
            output_path = output_dir / f"{analysis_id}.csv"
            # Convert results to DataFrame and save
            df = pd.DataFrame([results])
            df.to_csv(output_path, index=False)
        else:
            output_path = output_dir / f"{analysis_id}.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        return str(output_path)
    
    async def _analyze_trends(self, request: TrendAnalysisRequest) -> HistoricalAnalysisResponse:
        """Analyze trends in historical data."""
        analysis_request = HistoricalAnalysisRequest(
            analysis_type="trend",
            start_date=request.start_date,
            end_date=request.end_date,
            data_sources=request.data_sources,
            metrics=request.metrics,
            aggregation_period="daily",
            output_format="json"
        )
        
        return await self._perform_historical_analysis(analysis_request)
    
    async def _analyze_performance(self, request: PerformanceAnalysisRequest) -> HistoricalAnalysisResponse:
        """Analyze performance over time."""
        analysis_request = HistoricalAnalysisRequest(
            analysis_type="performance",
            start_date=request.start_date,
            end_date=request.end_date,
            data_sources=request.data_sources,
            metrics=request.performance_metrics,
            aggregation_period="daily",
            output_format="json"
        )
        
        return await self._perform_historical_analysis(analysis_request)
    
    async def _analyze_quality(self, request: QualityAnalysisRequest) -> HistoricalAnalysisResponse:
        """Analyze quality trends."""
        analysis_request = HistoricalAnalysisRequest(
            analysis_type="quality",
            start_date=request.start_date,
            end_date=request.end_date,
            data_sources=request.data_sources,
            metrics=request.quality_metrics,
            aggregation_period="daily",
            output_format="json"
        )
        
        return await self._perform_historical_analysis(analysis_request)
    
    async def _list_analyses(self, limit: int) -> Dict[str, Any]:
        """List recent analyses."""
        # This would implement actual listing logic
        # For now, return mock data
        return {
            'analyses': [],
            'total_analyses': 0,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_analysis(self, analysis_id: str) -> Dict[str, Any]:
        """Get analysis results."""
        # This would implement actual retrieval logic
        # For now, return mock data
        return {
            'analysis_id': analysis_id,
            'status': 'not_found',
            'message': 'Analysis not found'
        }
    
    def run(self, host: str = "0.0.0.0", port: int = 8007):
        """Run the service."""
        logger.info(f"Starting Historical Analysis Service on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# Global service instance
service = HistoricalAnalysisService()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    return service.app


if __name__ == "__main__":
    service.run()
