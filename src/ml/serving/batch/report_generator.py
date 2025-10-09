"""
Report Generator Service

This module implements the report generator service for PBF-LB/M processes.
It provides REST API endpoints for generating comprehensive reports,
automated reporting, and report scheduling.
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
from jinja2 import Template
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

from ...pipelines.inference.batch_inference import BatchInferencePipeline
from ...models.base_model import BaseModel
from ...config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


# Pydantic models for API requests and responses
class ReportRequest(BaseModel):
    """Request model for report generation."""
    report_type: str = Field(..., description="Type of report (performance, quality, defects, summary)")
    report_name: str = Field(..., description="Name of the report")
    start_date: str = Field(..., description="Start date for report data (ISO format)")
    end_date: str = Field(..., description="End date for report data (ISO format)")
    data_sources: List[str] = Field(..., description="List of data sources to include")
    sections: List[str] = Field(..., description="Report sections to include")
    output_format: str = Field("pdf", description="Output format (pdf, html, excel, json)")
    template: Optional[str] = Field(None, description="Custom report template")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")
    include_charts: bool = Field(True, description="Include charts and visualizations")
    include_recommendations: bool = Field(True, description="Include recommendations")


class ReportResponse(BaseModel):
    """Response model for report generation."""
    report_id: str = Field(..., description="Report ID")
    report_name: str = Field(..., description="Report name")
    report_type: str = Field(..., description="Report type")
    status: str = Field(..., description="Report status")
    output_path: str = Field(..., description="Output file path")
    file_size: int = Field(..., description="File size in bytes")
    sections_included: List[str] = Field(..., description="Sections included in report")
    created_at: str = Field(..., description="Report creation timestamp")
    message: str = Field(..., description="Response message")


class ScheduledReportRequest(BaseModel):
    """Request model for scheduled report creation."""
    report_name: str = Field(..., description="Name of the scheduled report")
    report_type: str = Field(..., description="Type of report")
    schedule_type: str = Field(..., description="Schedule type (daily, weekly, monthly)")
    schedule_config: Dict[str, Any] = Field(..., description="Schedule configuration")
    data_sources: List[str] = Field(..., description="Data sources for the report")
    sections: List[str] = Field(..., description="Report sections to include")
    output_format: str = Field("pdf", description="Output format")
    recipients: List[str] = Field(..., description="Report recipients")
    template: Optional[str] = Field(None, description="Custom report template")
    enabled: bool = Field(True, description="Whether the scheduled report is enabled")


class ReportTemplate(BaseModel):
    """Model for report template."""
    template_id: str = Field(..., description="Template ID")
    template_name: str = Field(..., description="Template name")
    template_type: str = Field(..., description="Template type")
    template_content: str = Field(..., description="Template content")
    variables: List[str] = Field(..., description="Template variables")
    created_at: str = Field(..., description="Template creation timestamp")


class ReportGeneratorService:
    """
    Report generator service for PBF-LB/M processes.
    
    This service provides report generation capabilities for:
    - Performance reports
    - Quality reports
    - Defect analysis reports
    - Summary reports
    - Custom reports
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the report generator service.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.app = FastAPI(
            title="Report Generator Service",
            description="Report generation for PBF-LB/M manufacturing",
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
        
        # Report counter
        self.report_counter = 0
        
        # Service metrics
        self.service_metrics = {
            'total_reports': 0,
            'performance_reports': 0,
            'quality_reports': 0,
            'defect_reports': 0,
            'summary_reports': 0,
            'last_report_time': None
        }
        
        # Report templates
        self.templates = self._load_default_templates()
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Initialized ReportGeneratorService")
    
    def _load_default_templates(self) -> Dict[str, ReportTemplate]:
        """Load default report templates."""
        templates = {}
        
        # Performance report template
        performance_template = """
# Performance Report - {{ report_name }}

## Executive Summary
- **Report Period**: {{ start_date }} to {{ end_date }}
- **Total Records**: {{ total_records }}
- **Overall Performance Score**: {{ overall_performance }}

## Key Metrics
{% for metric, value in key_metrics.items() %}
- **{{ metric }}**: {{ value }}
{% endfor %}

## Performance Trends
{{ performance_trends }}

## Recommendations
{% for recommendation in recommendations %}
- {{ recommendation }}
{% endfor %}
"""
        
        templates['performance'] = ReportTemplate(
            template_id='performance_default',
            template_name='Default Performance Report',
            template_type='performance',
            template_content=performance_template,
            variables=['report_name', 'start_date', 'end_date', 'total_records', 'overall_performance', 'key_metrics', 'performance_trends', 'recommendations'],
            created_at=datetime.now().isoformat()
        )
        
        # Quality report template
        quality_template = """
# Quality Report - {{ report_name }}

## Quality Summary
- **Report Period**: {{ start_date }} to {{ end_date }}
- **Overall Quality Score**: {{ overall_quality }}
- **Defect Rate**: {{ defect_rate }}%

## Quality Metrics
{% for metric, value in quality_metrics.items() %}
- **{{ metric }}**: {{ value }}
{% endfor %}

## Quality Trends
{{ quality_trends }}

## Defect Analysis
{{ defect_analysis }}

## Recommendations
{% for recommendation in recommendations %}
- {{ recommendation }}
{% endfor %}
"""
        
        templates['quality'] = ReportTemplate(
            template_id='quality_default',
            template_name='Default Quality Report',
            template_type='quality',
            template_content=quality_template,
            variables=['report_name', 'start_date', 'end_date', 'overall_quality', 'defect_rate', 'quality_metrics', 'quality_trends', 'defect_analysis', 'recommendations'],
            created_at=datetime.now().isoformat()
        )
        
        return templates
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": "report_generator",
                "timestamp": datetime.now().isoformat(),
                "metrics": self.service_metrics
            }
        
        @self.app.post("/generate", response_model=ReportResponse)
        async def generate_report(request: ReportRequest):
            """Generate a report."""
            return await self._generate_report(request)
        
        @self.app.post("/schedule", response_model=ReportResponse)
        async def schedule_report(request: ScheduledReportRequest):
            """Schedule a report."""
            return await self._schedule_report(request)
        
        @self.app.get("/templates")
        async def list_templates():
            """List available report templates."""
            return await self._list_templates()
        
        @self.app.get("/templates/{template_id}")
        async def get_template(template_id: str):
            """Get a specific template."""
            return await self._get_template(template_id)
        
        @self.app.post("/templates")
        async def create_template(template: ReportTemplate):
            """Create a new template."""
            return await self._create_template(template)
        
        @self.app.get("/reports")
        async def list_reports(limit: int = Query(10, ge=1, le=100)):
            """List recent reports."""
            return await self._list_reports(limit)
        
        @self.app.get("/reports/{report_id}")
        async def get_report(report_id: str):
            """Get report information."""
            return await self._get_report(report_id)
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get service metrics."""
            return self.service_metrics
    
    async def _generate_report(self, request: ReportRequest) -> ReportResponse:
        """
        Generate a report.
        
        Args:
            request: Report generation request
            
        Returns:
            Report response
        """
        # Generate report ID
        self.report_counter += 1
        report_id = f"report_{self.report_counter}_{int(time.time())}"
        
        try:
            # Load data for the report
            report_data = await self._load_report_data(
                start_date=request.start_date,
                end_date=request.end_date,
                data_sources=request.data_sources,
                filters=request.filters
            )
            
            # Generate report content
            report_content = await self._generate_report_content(
                report_data, request, report_id
            )
            
            # Generate visualizations if requested
            visualizations = {}
            if request.include_charts:
                visualizations = await self._generate_report_visualizations(
                    report_data, request
                )
            
            # Generate recommendations if requested
            recommendations = []
            if request.include_recommendations:
                recommendations = await self._generate_recommendations(
                    report_data, request
                )
            
            # Save report
            output_path = await self._save_report(
                report_id, report_content, visualizations, request
            )
            
            # Update metrics
            self.service_metrics['total_reports'] += 1
            self.service_metrics[f'{request.report_type}_reports'] += 1
            self.service_metrics['last_report_time'] = datetime.now().isoformat()
            
            return ReportResponse(
                report_id=report_id,
                report_name=request.report_name,
                report_type=request.report_type,
                status='completed',
                output_path=output_path,
                file_size=Path(output_path).stat().st_size if Path(output_path).exists() else 0,
                sections_included=request.sections,
                created_at=datetime.now().isoformat(),
                message="Report generated successfully"
            )
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _load_report_data(self, start_date: str, end_date: str, 
                              data_sources: List[str], filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load data for the report."""
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
            'build_time': np.random.normal(3600, 300, len(date_range)),
            'energy_consumption': np.random.normal(50, 5, len(date_range)),
            'material_usage': np.random.normal(100, 10, len(date_range))
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
        
        return {
            'dataframe': df,
            'summary_stats': df.describe().to_dict(),
            'total_records': len(df),
            'date_range': {'start': start_date, 'end': end_date}
        }
    
    async def _generate_report_content(self, report_data: Dict[str, Any], 
                                     request: ReportRequest, report_id: str) -> str:
        """Generate report content using template."""
        # Get template
        template = self.templates.get(request.report_type)
        if not template:
            # Use default template
            template = self.templates['performance']
        
        # Prepare template variables
        template_vars = {
            'report_name': request.report_name,
            'start_date': request.start_date,
            'end_date': request.end_date,
            'total_records': report_data['total_records'],
            'report_id': report_id
        }
        
        # Add type-specific variables
        if request.report_type == 'performance':
            template_vars.update(await self._prepare_performance_variables(report_data))
        elif request.report_type == 'quality':
            template_vars.update(await self._prepare_quality_variables(report_data))
        elif request.report_type == 'defects':
            template_vars.update(await self._prepare_defect_variables(report_data))
        elif request.report_type == 'summary':
            template_vars.update(await self._prepare_summary_variables(report_data))
        
        # Render template
        jinja_template = Template(template.template_content)
        content = jinja_template.render(**template_vars)
        
        return content
    
    async def _prepare_performance_variables(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare variables for performance report."""
        df = report_data['dataframe']
        
        # Calculate performance metrics
        performance_metrics = {
            'mean_build_time': float(df['build_time'].mean()),
            'mean_energy_consumption': float(df['energy_consumption'].mean()),
            'mean_material_usage': float(df['material_usage'].mean()),
            'efficiency_score': float(1 - (df['build_time'].std() / df['build_time'].mean())),
            'energy_efficiency': float(1 - (df['energy_consumption'].std() / df['energy_consumption'].mean()))
        }
        
        # Calculate overall performance
        overall_performance = np.mean([
            performance_metrics['efficiency_score'],
            performance_metrics['energy_efficiency']
        ])
        
        # Performance trends
        performance_trends = "Performance has been stable over the reporting period with minor fluctuations."
        
        return {
            'key_metrics': performance_metrics,
            'overall_performance': float(overall_performance),
            'performance_trends': performance_trends
        }
    
    async def _prepare_quality_variables(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare variables for quality report."""
        df = report_data['dataframe']
        
        # Calculate quality metrics
        quality_metrics = {
            'mean_quality_score': float(df['quality_score'].mean()),
            'quality_consistency': float(1 - df['quality_score'].std()),
            'high_quality_percentage': float(np.sum(df['quality_score'] > 0.8) / len(df) * 100),
            'low_quality_percentage': float(np.sum(df['quality_score'] < 0.6) / len(df) * 100)
        }
        
        # Calculate defect rate
        defect_rate = float(np.sum(df['defect_count'] > 0) / len(df) * 100)
        
        # Quality trends
        quality_trends = "Quality scores have remained consistent with slight improvements over time."
        
        # Defect analysis
        defect_analysis = f"Total defects: {int(df['defect_count'].sum())}, Mean defects per unit: {float(df['defect_count'].mean()):.2f}"
        
        return {
            'quality_metrics': quality_metrics,
            'overall_quality': float(df['quality_score'].mean()),
            'defect_rate': defect_rate,
            'quality_trends': quality_trends,
            'defect_analysis': defect_analysis
        }
    
    async def _prepare_defect_variables(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare variables for defect report."""
        df = report_data['dataframe']
        
        # Defect metrics
        defect_metrics = {
            'total_defects': int(df['defect_count'].sum()),
            'mean_defects_per_unit': float(df['defect_count'].mean()),
            'max_defects': int(df['defect_count'].max()),
            'defect_frequency': float(np.sum(df['defect_count'] > 0) / len(df) * 100),
            'zero_defect_percentage': float(np.sum(df['defect_count'] == 0) / len(df) * 100)
        }
        
        # Defect trends
        defect_trends = "Defect rates have been relatively stable with occasional spikes."
        
        return {
            'defect_metrics': defect_metrics,
            'defect_trends': defect_trends
        }
    
    async def _prepare_summary_variables(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare variables for summary report."""
        df = report_data['dataframe']
        
        # Summary metrics
        summary_metrics = {
            'total_records': len(df),
            'mean_quality_score': float(df['quality_score'].mean()),
            'total_defects': int(df['defect_count'].sum()),
            'mean_build_time': float(df['build_time'].mean()),
            'mean_energy_consumption': float(df['energy_consumption'].mean())
        }
        
        # Overall summary
        overall_summary = "The manufacturing process has been operating within normal parameters."
        
        return {
            'summary_metrics': summary_metrics,
            'overall_summary': overall_summary
        }
    
    async def _generate_report_visualizations(self, report_data: Dict[str, Any], 
                                            request: ReportRequest) -> Dict[str, Any]:
        """Generate visualizations for the report."""
        visualizations = {}
        df = report_data['dataframe']
        
        try:
            # Time series plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['quality_score'],
                mode='lines',
                name='Quality Score',
                line=dict(width=2, color='blue')
            ))
            fig.update_layout(
                title="Quality Score Over Time",
                xaxis_title="Time",
                yaxis_title="Quality Score",
                hovermode='x unified'
            )
            visualizations['quality_trend'] = json.loads(fig.to_json())
            
            # Defect distribution
            fig = px.histogram(df, x='defect_count', title="Defect Count Distribution")
            visualizations['defect_distribution'] = json.loads(fig.to_json())
            
            # Performance metrics
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['build_time'],
                mode='lines',
                name='Build Time',
                line=dict(width=2, color='green')
            ))
            fig.update_layout(
                title="Build Time Over Time",
                xaxis_title="Time",
                yaxis_title="Build Time (seconds)",
                hovermode='x unified'
            )
            visualizations['build_time_trend'] = json.loads(fig.to_json())
            
        except Exception as e:
            logger.warning(f"Could not generate visualizations: {e}")
        
        return visualizations
    
    async def _generate_recommendations(self, report_data: Dict[str, Any], 
                                      request: ReportRequest) -> List[str]:
        """Generate recommendations based on the data."""
        recommendations = []
        df = report_data['dataframe']
        
        # Quality recommendations
        if 'quality_score' in df.columns:
            mean_quality = df['quality_score'].mean()
            if mean_quality < 0.8:
                recommendations.append("Consider optimizing process parameters to improve quality scores")
            if df['quality_score'].std() > 0.1:
                recommendations.append("Focus on process stability to reduce quality score variability")
        
        # Defect recommendations
        if 'defect_count' in df.columns:
            defect_rate = np.sum(df['defect_count'] > 0) / len(df)
            if defect_rate > 0.1:
                recommendations.append("Implement additional quality control measures to reduce defect rate")
        
        # Performance recommendations
        if 'build_time' in df.columns:
            if df['build_time'].std() / df['build_time'].mean() > 0.2:
                recommendations.append("Optimize build parameters to reduce build time variability")
        
        # Energy recommendations
        if 'energy_consumption' in df.columns:
            if df['energy_consumption'].mean() > 60:
                recommendations.append("Review energy consumption patterns and optimize for efficiency")
        
        return recommendations
    
    async def _save_report(self, report_id: str, content: str, visualizations: Dict[str, Any], 
                          request: ReportRequest) -> str:
        """Save the report to file."""
        output_dir = Path("outputs/reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if request.output_format == "html":
            output_path = output_dir / f"{report_id}.html"
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{request.report_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1 {{ color: #333; }}
                    h2 {{ color: #666; }}
                    .chart {{ margin: 20px 0; }}
                </style>
            </head>
            <body>
                {content}
                <div class="chart">
                    <h3>Visualizations</h3>
                    {json.dumps(visualizations)}
                </div>
            </body>
            </html>
            """
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        elif request.output_format == "json":
            output_path = output_dir / f"{report_id}.json"
            report_data = {
                'content': content,
                'visualizations': visualizations,
                'metadata': {
                    'report_id': report_id,
                    'report_name': request.report_name,
                    'report_type': request.report_type,
                    'created_at': datetime.now().isoformat()
                }
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, default=str)
        else:
            # Default to text format
            output_path = output_dir / f"{report_id}.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        return str(output_path)
    
    async def _schedule_report(self, request: ScheduledReportRequest) -> ReportResponse:
        """Schedule a report."""
        # This would implement report scheduling logic
        # For now, return a mock response
        self.report_counter += 1
        report_id = f"scheduled_report_{self.report_counter}_{int(time.time())}"
        
        return ReportResponse(
            report_id=report_id,
            report_name=request.report_name,
            report_type=request.report_type,
            status='scheduled',
            output_path="",
            file_size=0,
            sections_included=request.sections,
            created_at=datetime.now().isoformat(),
            message="Report scheduled successfully"
        )
    
    async def _list_templates(self) -> Dict[str, Any]:
        """List available report templates."""
        template_list = []
        for template_id, template in self.templates.items():
            template_list.append({
                'template_id': template_id,
                'template_name': template.template_name,
                'template_type': template.template_type,
                'variables': template.variables,
                'created_at': template.created_at
            })
        
        return {
            'templates': template_list,
            'total_templates': len(template_list),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_template(self, template_id: str) -> ReportTemplate:
        """Get a specific template."""
        if template_id not in self.templates:
            raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
        
        return self.templates[template_id]
    
    async def _create_template(self, template: ReportTemplate) -> Dict[str, Any]:
        """Create a new template."""
        self.templates[template.template_id] = template
        
        return {
            'template_id': template.template_id,
            'status': 'created',
            'message': 'Template created successfully',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _list_reports(self, limit: int) -> Dict[str, Any]:
        """List recent reports."""
        # This would implement actual listing logic
        # For now, return mock data
        return {
            'reports': [],
            'total_reports': 0,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_report(self, report_id: str) -> Dict[str, Any]:
        """Get report information."""
        # This would implement actual retrieval logic
        # For now, return mock data
        return {
            'report_id': report_id,
            'status': 'not_found',
            'message': 'Report not found'
        }
    
    def run(self, host: str = "0.0.0.0", port: int = 8008):
        """Run the service."""
        logger.info(f"Starting Report Generator Service on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# Global service instance
service = ReportGeneratorService()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    return service.app


if __name__ == "__main__":
    service.run()
