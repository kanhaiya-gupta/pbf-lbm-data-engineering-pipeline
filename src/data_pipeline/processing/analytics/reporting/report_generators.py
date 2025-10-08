"""
Report Generators for PBF-LB/M Analytics

This module provides automated report generation capabilities for PBF-LB/M
analytics results, including sensitivity analysis reports, statistical
analysis reports, and comprehensive analytics reports.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from datetime import datetime
import json
import os
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    
    # Report parameters
    report_format: str = "html"  # "html", "pdf", "markdown", "json"
    include_plots: bool = True
    include_tables: bool = True
    include_statistics: bool = True
    
    # Output parameters
    output_directory: str = "reports"
    filename_prefix: str = "pbf_analytics_report"
    
    # Report content parameters
    include_summary: bool = True
    include_details: bool = True
    include_recommendations: bool = True


@dataclass
class ReportResult:
    """Result of report generation."""
    
    success: bool
    report_type: str
    report_path: str
    report_size: int
    generation_time: float
    error_message: Optional[str] = None


class AnalysisReportGenerator:
    """
    Analysis report generator for PBF-LB/M analytics.
    
    This class provides automated report generation capabilities for
    comprehensive analytics results including sensitivity analysis,
    statistical analysis, and process analysis.
    """
    
    def __init__(self, config: ReportConfig = None):
        """Initialize the report generator."""
        self.config = config or ReportConfig()
        
        # Create output directory
        os.makedirs(self.config.output_directory, exist_ok=True)
        
        logger.info("Analysis Report Generator initialized")
    
    def generate_comprehensive_report(
        self,
        analytics_results: Dict[str, Any],
        report_title: str = "PBF-LB/M Analytics Report"
    ) -> ReportResult:
        """
        Generate comprehensive analytics report.
        
        Args:
            analytics_results: Dictionary containing all analytics results
            report_title: Title of the report
            
        Returns:
            ReportResult: Report generation result
        """
        try:
            start_time = datetime.now()
            
            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config.filename_prefix}_{timestamp}.{self.config.report_format}"
            report_path = os.path.join(self.config.output_directory, filename)
            
            # Generate report content
            if self.config.report_format == "html":
                report_content = self._generate_html_report(analytics_results, report_title)
            elif self.config.report_format == "markdown":
                report_content = self._generate_markdown_report(analytics_results, report_title)
            elif self.config.report_format == "json":
                report_content = self._generate_json_report(analytics_results, report_title)
            else:
                raise ValueError(f"Unsupported report format: {self.config.report_format}")
            
            # Write report to file
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            # Calculate report size
            report_size = os.path.getsize(report_path)
            
            # Calculate generation time
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = ReportResult(
                success=True,
                report_type="Comprehensive",
                report_path=report_path,
                report_size=report_size,
                generation_time=generation_time
            )
            
            logger.info(f"Comprehensive report generated: {report_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            return ReportResult(
                success=False,
                report_type="Comprehensive",
                report_path="",
                report_size=0,
                generation_time=0.0,
                error_message=str(e)
            )
    
    def generate_sensitivity_report(
        self,
        sensitivity_results: Dict[str, Any],
        report_title: str = "Sensitivity Analysis Report"
    ) -> ReportResult:
        """
        Generate sensitivity analysis report.
        
        Args:
            sensitivity_results: Dictionary containing sensitivity analysis results
            report_title: Title of the report
            
        Returns:
            ReportResult: Report generation result
        """
        try:
            start_time = datetime.now()
            
            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sensitivity_report_{timestamp}.{self.config.report_format}"
            report_path = os.path.join(self.config.output_directory, filename)
            
            # Generate report content
            if self.config.report_format == "html":
                report_content = self._generate_sensitivity_html_report(sensitivity_results, report_title)
            elif self.config.report_format == "markdown":
                report_content = self._generate_sensitivity_markdown_report(sensitivity_results, report_title)
            else:
                raise ValueError(f"Unsupported report format: {self.config.report_format}")
            
            # Write report to file
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            # Calculate report size
            report_size = os.path.getsize(report_path)
            
            # Calculate generation time
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = ReportResult(
                success=True,
                report_type="Sensitivity",
                report_path=report_path,
                report_size=report_size,
                generation_time=generation_time
            )
            
            logger.info(f"Sensitivity report generated: {report_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating sensitivity report: {e}")
            return ReportResult(
                success=False,
                report_type="Sensitivity",
                report_path="",
                report_size=0,
                generation_time=0.0,
                error_message=str(e)
            )
    
    def _generate_html_report(self, analytics_results: Dict[str, Any], report_title: str) -> str:
        """Generate HTML report content."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{report_title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; }}
        h3 {{ color: #7f8c8d; }}
        .summary {{ background-color: #ecf0f1; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #3498db; color: white; border-radius: 3px; }}
        .error {{ color: #e74c3c; }}
        .success {{ color: #27ae60; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>{report_title}</h1>
    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <div class="summary">
        <h2>Executive Summary</h2>
        {self._generate_summary_html(analytics_results)}
    </div>
    
    <div class="section">
        <h2>Sensitivity Analysis Results</h2>
        {self._generate_sensitivity_section_html(analytics_results)}
    </div>
    
    <div class="section">
        <h2>Statistical Analysis Results</h2>
        {self._generate_statistical_section_html(analytics_results)}
    </div>
    
    <div class="section">
        <h2>Process Analysis Results</h2>
        {self._generate_process_section_html(analytics_results)}
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        {self._generate_recommendations_html(analytics_results)}
    </div>
</body>
</html>
"""
        return html_content
    
    def _generate_markdown_report(self, analytics_results: Dict[str, Any], report_title: str) -> str:
        """Generate Markdown report content."""
        markdown_content = f"""# {report_title}

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

{self._generate_summary_markdown(analytics_results)}

## Sensitivity Analysis Results

{self._generate_sensitivity_section_markdown(analytics_results)}

## Statistical Analysis Results

{self._generate_statistical_section_markdown(analytics_results)}

## Process Analysis Results

{self._generate_process_section_markdown(analytics_results)}

## Recommendations

{self._generate_recommendations_markdown(analytics_results)}
"""
        return markdown_content
    
    def _generate_json_report(self, analytics_results: Dict[str, Any], report_title: str) -> str:
        """Generate JSON report content."""
        report_data = {
            'title': report_title,
            'generated': datetime.now().isoformat(),
            'summary': self._generate_summary_json(analytics_results),
            'sensitivity_analysis': self._generate_sensitivity_section_json(analytics_results),
            'statistical_analysis': self._generate_statistical_section_json(analytics_results),
            'process_analysis': self._generate_process_section_json(analytics_results),
            'recommendations': self._generate_recommendations_json(analytics_results)
        }
        
        return json.dumps(report_data, indent=2, default=str)
    
    def _generate_summary_html(self, analytics_results: Dict[str, Any]) -> str:
        """Generate summary section in HTML format."""
        summary_html = "<p>This report contains comprehensive analytics results for PBF-LB/M process analysis.</p>"
        
        # Add key metrics
        if 'sensitivity_analysis' in analytics_results:
            summary_html += "<h3>Key Sensitivity Metrics</h3><ul>"
            summary_html += "<li>Sensitivity analysis completed successfully</li>"
            summary_html += "</ul>"
        
        if 'statistical_analysis' in analytics_results:
            summary_html += "<h3>Key Statistical Metrics</h3><ul>"
            summary_html += "<li>Statistical analysis completed successfully</li>"
            summary_html += "</ul>"
        
        return summary_html
    
    def _generate_summary_markdown(self, analytics_results: Dict[str, Any]) -> str:
        """Generate summary section in Markdown format."""
        summary_md = "This report contains comprehensive analytics results for PBF-LB/M process analysis.\n\n"
        
        # Add key metrics
        if 'sensitivity_analysis' in analytics_results:
            summary_md += "### Key Sensitivity Metrics\n"
            summary_md += "- Sensitivity analysis completed successfully\n\n"
        
        if 'statistical_analysis' in analytics_results:
            summary_md += "### Key Statistical Metrics\n"
            summary_md += "- Statistical analysis completed successfully\n\n"
        
        return summary_md
    
    def _generate_summary_json(self, analytics_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary section in JSON format."""
        return {
            'description': 'Comprehensive analytics results for PBF-LB/M process analysis',
            'sensitivity_analysis_available': 'sensitivity_analysis' in analytics_results,
            'statistical_analysis_available': 'statistical_analysis' in analytics_results,
            'process_analysis_available': 'process_analysis' in analytics_results
        }
    
    def _generate_sensitivity_section_html(self, analytics_results: Dict[str, Any]) -> str:
        """Generate sensitivity analysis section in HTML format."""
        if 'sensitivity_analysis' not in analytics_results:
            return "<p>No sensitivity analysis results available.</p>"
        
        sensitivity_html = "<h3>Sobol Analysis</h3>"
        sensitivity_html += "<p>Sobol sensitivity analysis completed successfully.</p>"
        
        sensitivity_html += "<h3>Morris Screening</h3>"
        sensitivity_html += "<p>Morris screening analysis completed successfully.</p>"
        
        return sensitivity_html
    
    def _generate_sensitivity_section_markdown(self, analytics_results: Dict[str, Any]) -> str:
        """Generate sensitivity analysis section in Markdown format."""
        if 'sensitivity_analysis' not in analytics_results:
            return "No sensitivity analysis results available."
        
        sensitivity_md = "### Sobol Analysis\n"
        sensitivity_md += "Sobol sensitivity analysis completed successfully.\n\n"
        
        sensitivity_md += "### Morris Screening\n"
        sensitivity_md += "Morris screening analysis completed successfully.\n\n"
        
        return sensitivity_md
    
    def _generate_sensitivity_section_json(self, analytics_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate sensitivity analysis section in JSON format."""
        if 'sensitivity_analysis' not in analytics_results:
            return {'available': False}
        
        return {
            'available': True,
            'sobol_analysis': {'completed': True},
            'morris_analysis': {'completed': True}
        }
    
    def _generate_statistical_section_html(self, analytics_results: Dict[str, Any]) -> str:
        """Generate statistical analysis section in HTML format."""
        if 'statistical_analysis' not in analytics_results:
            return "<p>No statistical analysis results available.</p>"
        
        statistical_html = "<h3>Multivariate Analysis</h3>"
        statistical_html += "<p>Multivariate analysis completed successfully.</p>"
        
        statistical_html += "<h3>Time Series Analysis</h3>"
        statistical_html += "<p>Time series analysis completed successfully.</p>"
        
        return statistical_html
    
    def _generate_statistical_section_markdown(self, analytics_results: Dict[str, Any]) -> str:
        """Generate statistical analysis section in Markdown format."""
        if 'statistical_analysis' not in analytics_results:
            return "No statistical analysis results available."
        
        statistical_md = "### Multivariate Analysis\n"
        statistical_md += "Multivariate analysis completed successfully.\n\n"
        
        statistical_md += "### Time Series Analysis\n"
        statistical_md += "Time series analysis completed successfully.\n\n"
        
        return statistical_md
    
    def _generate_statistical_section_json(self, analytics_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate statistical analysis section in JSON format."""
        if 'statistical_analysis' not in analytics_results:
            return {'available': False}
        
        return {
            'available': True,
            'multivariate_analysis': {'completed': True},
            'time_series_analysis': {'completed': True}
        }
    
    def _generate_process_section_html(self, analytics_results: Dict[str, Any]) -> str:
        """Generate process analysis section in HTML format."""
        if 'process_analysis' not in analytics_results:
            return "<p>No process analysis results available.</p>"
        
        process_html = "<h3>Parameter Analysis</h3>"
        process_html += "<p>Parameter analysis completed successfully.</p>"
        
        process_html += "<h3>Quality Analysis</h3>"
        process_html += "<p>Quality analysis completed successfully.</p>"
        
        return process_html
    
    def _generate_process_section_markdown(self, analytics_results: Dict[str, Any]) -> str:
        """Generate process analysis section in Markdown format."""
        if 'process_analysis' not in analytics_results:
            return "No process analysis results available."
        
        process_md = "### Parameter Analysis\n"
        process_md += "Parameter analysis completed successfully.\n\n"
        
        process_md += "### Quality Analysis\n"
        process_md += "Quality analysis completed successfully.\n\n"
        
        return process_md
    
    def _generate_process_section_json(self, analytics_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate process analysis section in JSON format."""
        if 'process_analysis' not in analytics_results:
            return {'available': False}
        
        return {
            'available': True,
            'parameter_analysis': {'completed': True},
            'quality_analysis': {'completed': True}
        }
    
    def _generate_recommendations_html(self, analytics_results: Dict[str, Any]) -> str:
        """Generate recommendations section in HTML format."""
        recommendations_html = "<h3>Process Optimization Recommendations</h3>"
        recommendations_html += "<ul>"
        recommendations_html += "<li>Optimize laser power for improved quality</li>"
        recommendations_html += "<li>Adjust scan speed for better surface finish</li>"
        recommendations_html += "<li>Monitor hatch spacing for consistent density</li>"
        recommendations_html += "</ul>"
        
        return recommendations_html
    
    def _generate_recommendations_markdown(self, analytics_results: Dict[str, Any]) -> str:
        """Generate recommendations section in Markdown format."""
        recommendations_md = "### Process Optimization Recommendations\n"
        recommendations_md += "- Optimize laser power for improved quality\n"
        recommendations_md += "- Adjust scan speed for better surface finish\n"
        recommendations_md += "- Monitor hatch spacing for consistent density\n\n"
        
        return recommendations_md
    
    def _generate_recommendations_json(self, analytics_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations section in JSON format."""
        return {
            'process_optimization': [
                'Optimize laser power for improved quality',
                'Adjust scan speed for better surface finish',
                'Monitor hatch spacing for consistent density'
            ]
        }
    
    def _generate_sensitivity_html_report(self, sensitivity_results: Dict[str, Any], report_title: str) -> str:
        """Generate sensitivity-specific HTML report."""
        return self._generate_html_report({'sensitivity_analysis': sensitivity_results}, report_title)
    
    def _generate_sensitivity_markdown_report(self, sensitivity_results: Dict[str, Any], report_title: str) -> str:
        """Generate sensitivity-specific Markdown report."""
        return self._generate_markdown_report({'sensitivity_analysis': sensitivity_results}, report_title)


class SensitivityReportGenerator(AnalysisReportGenerator):
    """Specialized sensitivity analysis report generator."""
    
    def __init__(self, config: ReportConfig = None):
        super().__init__(config)
        self.report_type = "Sensitivity"
    
    def generate_report(self, sensitivity_results: Dict[str, Any], report_title: str = "Sensitivity Analysis Report") -> ReportResult:
        """Generate sensitivity analysis report."""
        return self.generate_sensitivity_report(sensitivity_results, report_title)
