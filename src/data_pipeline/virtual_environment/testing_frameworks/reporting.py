"""
Reporting for PBF-LB/M Virtual Environment Testing

This module provides reporting capabilities for virtual environment testing including
test report generation, test visualization, and comprehensive test documentation
for PBF-LB/M virtual testing and simulation environments.
"""

import json
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import uuid
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
import warnings

logger = logging.getLogger(__name__)


class ReportType(Enum):
    """Report type enumeration."""
    TEST_SUMMARY = "test_summary"
    DETAILED_REPORT = "detailed_report"
    PERFORMANCE_REPORT = "performance_report"
    VALIDATION_REPORT = "validation_report"
    COMPARISON_REPORT = "comparison_report"
    TREND_REPORT = "trend_report"


class ReportFormat(Enum):
    """Report format enumeration."""
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    MARKDOWN = "markdown"


@dataclass
class ReportConfig:
    """Report configuration."""
    
    report_id: str
    name: str
    report_type: ReportType
    format: ReportFormat
    created_at: datetime
    updated_at: datetime
    
    # Report parameters
    include_charts: bool = True
    include_details: bool = True
    include_recommendations: bool = True
    
    # Output parameters
    output_directory: str = "./reports"
    template_path: Optional[str] = None


@dataclass
class ReportData:
    """Report data structure."""
    
    report_id: str
    timestamp: datetime
    report_type: ReportType
    
    # Summary data
    summary: Dict[str, Any]
    
    # Detailed data
    details: Dict[str, Any]
    
    # Charts data
    charts: List[Dict[str, Any]]
    
    # Recommendations
    recommendations: List[str]


class TestReportGenerator:
    """
    Test report generator for PBF-LB/M virtual environment.
    
    This class provides comprehensive test reporting capabilities including
    test summary reports, detailed reports, and performance reports for
    PBF-LB/M virtual testing and simulation environments.
    """
    
    def __init__(self):
        """Initialize the test report generator."""
        self.report_configs = {}
        self.report_templates = {}
        self.generated_reports = {}
        
        # Initialize default templates
        self._initialize_default_templates()
        
        logger.info("Test Report Generator initialized")
    
    async def create_report_config(
        self,
        name: str,
        report_type: ReportType,
        format: ReportFormat = ReportFormat.HTML,
        output_directory: str = "./reports"
    ) -> str:
        """
        Create report configuration.
        
        Args:
            name: Report name
            report_type: Type of report
            format: Report format
            output_directory: Output directory
            
        Returns:
            str: Report configuration ID
        """
        try:
            report_id = str(uuid.uuid4())
            
            config = ReportConfig(
                report_id=report_id,
                name=name,
                report_type=report_type,
                format=format,
                output_directory=output_directory,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            self.report_configs[report_id] = config
            
            logger.info(f"Report configuration created: {report_id}")
            return report_id
            
        except Exception as e:
            logger.error(f"Error creating report configuration: {e}")
            return ""
    
    async def generate_report(
        self,
        report_id: str,
        test_data: Dict[str, Any],
        validation_data: Dict[str, Any] = None,
        comparison_data: Dict[str, Any] = None
    ) -> str:
        """
        Generate test report.
        
        Args:
            report_id: Report configuration ID
            test_data: Test execution data
            validation_data: Validation data
            comparison_data: Comparison data
            
        Returns:
            str: Generated report file path
        """
        try:
            if report_id not in self.report_configs:
                raise ValueError(f"Report configuration not found: {report_id}")
            
            config = self.report_configs[report_id]
            
            # Prepare report data
            report_data = await self._prepare_report_data(
                config, test_data, validation_data, comparison_data
            )
            
            # Generate report based on format
            if config.format == ReportFormat.HTML:
                report_path = await self._generate_html_report(config, report_data)
            elif config.format == ReportFormat.JSON:
                report_path = await self._generate_json_report(config, report_data)
            elif config.format == ReportFormat.CSV:
                report_path = await self._generate_csv_report(config, report_data)
            elif config.format == ReportFormat.EXCEL:
                report_path = await self._generate_excel_report(config, report_data)
            elif config.format == ReportFormat.MARKDOWN:
                report_path = await self._generate_markdown_report(config, report_data)
            else:
                raise ValueError(f"Unsupported report format: {config.format}")
            
            # Store generated report
            self.generated_reports[report_id] = {
                'report_path': report_path,
                'generated_at': datetime.now(),
                'report_data': report_data
            }
            
            logger.info(f"Report generated: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return ""
    
    async def _prepare_report_data(
        self,
        config: ReportConfig,
        test_data: Dict[str, Any],
        validation_data: Dict[str, Any] = None,
        comparison_data: Dict[str, Any] = None
    ) -> ReportData:
        """Prepare report data."""
        try:
            # Generate summary
            summary = await self._generate_summary(test_data, validation_data, comparison_data)
            
            # Generate details
            details = await self._generate_details(test_data, validation_data, comparison_data)
            
            # Generate charts
            charts = await self._generate_charts(test_data, validation_data, comparison_data)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(test_data, validation_data, comparison_data)
            
            return ReportData(
                report_id=config.report_id,
                timestamp=datetime.now(),
                report_type=config.report_type,
                summary=summary,
                details=details,
                charts=charts,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error preparing report data: {e}")
            return ReportData(
                report_id=config.report_id,
                timestamp=datetime.now(),
                report_type=config.report_type,
                summary={},
                details={},
                charts=[],
                recommendations=[]
            )
    
    async def _generate_summary(
        self,
        test_data: Dict[str, Any],
        validation_data: Dict[str, Any] = None,
        comparison_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate report summary."""
        try:
            summary = {
                'total_tests': 0,
                'passed_tests': 0,
                'failed_tests': 0,
                'skipped_tests': 0,
                'error_tests': 0,
                'success_rate': 0.0,
                'total_execution_time': 0.0,
                'average_execution_time': 0.0,
                'test_types': {},
                'performance_metrics': {},
                'validation_metrics': {}
            }
            
            # Process test data
            if 'test_results' in test_data:
                test_results = test_data['test_results']
                summary['total_tests'] = len(test_results)
                
                for result in test_results:
                    status = result.get('status', 'unknown')
                    if status == 'passed':
                        summary['passed_tests'] += 1
                    elif status == 'failed':
                        summary['failed_tests'] += 1
                    elif status == 'skipped':
                        summary['skipped_tests'] += 1
                    elif status == 'error':
                        summary['error_tests'] += 1
                    
                    # Execution time
                    execution_time = result.get('execution_time', 0.0)
                    summary['total_execution_time'] += execution_time
                
                # Calculate success rate
                if summary['total_tests'] > 0:
                    summary['success_rate'] = summary['passed_tests'] / summary['total_tests']
                    summary['average_execution_time'] = summary['total_execution_time'] / summary['total_tests']
            
            # Process validation data
            if validation_data:
                summary['validation_metrics'] = {
                    'total_validations': len(validation_data.get('validation_results', [])),
                    'passed_validations': len([r for r in validation_data.get('validation_results', []) if r.get('status') == 'passed']),
                    'failed_validations': len([r for r in validation_data.get('validation_results', []) if r.get('status') == 'failed']),
                    'average_accuracy': np.mean([r.get('accuracy', 0.0) for r in validation_data.get('validation_results', [])]) if validation_data.get('validation_results') else 0.0
                }
            
            # Process comparison data
            if comparison_data:
                summary['comparison_metrics'] = {
                    'total_comparisons': len(comparison_data.get('comparison_results', [])),
                    'matching_comparisons': len([r for r in comparison_data.get('comparison_results', []) if r.get('overall_match', False)]),
                    'non_matching_comparisons': len([r for r in comparison_data.get('comparison_results', []) if not r.get('overall_match', False)])
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {}
    
    async def _generate_details(
        self,
        test_data: Dict[str, Any],
        validation_data: Dict[str, Any] = None,
        comparison_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate report details."""
        try:
            details = {
                'test_details': test_data,
                'validation_details': validation_data or {},
                'comparison_details': comparison_data or {},
                'execution_details': {},
                'error_details': []
            }
            
            # Extract execution details
            if 'test_results' in test_data:
                test_results = test_data['test_results']
                
                # Group by test type
                test_types = {}
                for result in test_results:
                    test_type = result.get('test_type', 'unknown')
                    if test_type not in test_types:
                        test_types[test_type] = []
                    test_types[test_type].append(result)
                
                details['execution_details']['test_types'] = test_types
                
                # Extract errors
                for result in test_results:
                    if result.get('status') in ['failed', 'error']:
                        error_detail = {
                            'test_id': result.get('test_id'),
                            'test_name': result.get('test_name'),
                            'error_message': result.get('error_message'),
                            'timestamp': result.get('timestamp')
                        }
                        details['error_details'].append(error_detail)
            
            return details
            
        except Exception as e:
            logger.error(f"Error generating details: {e}")
            return {}
    
    async def _generate_charts(
        self,
        test_data: Dict[str, Any],
        validation_data: Dict[str, Any] = None,
        comparison_data: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Generate charts data."""
        try:
            charts = []
            
            # Test results pie chart
            if 'test_results' in test_data:
                test_results = test_data['test_results']
                status_counts = {}
                for result in test_results:
                    status = result.get('status', 'unknown')
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                charts.append({
                    'type': 'pie',
                    'title': 'Test Results Distribution',
                    'data': status_counts,
                    'chart_id': 'test_results_pie'
                })
            
            # Execution time histogram
            if 'test_results' in test_data:
                test_results = test_data['test_results']
                execution_times = [result.get('execution_time', 0.0) for result in test_results]
                
                if execution_times:
                    charts.append({
                        'type': 'histogram',
                        'title': 'Execution Time Distribution',
                        'data': execution_times,
                        'chart_id': 'execution_time_hist'
                    })
            
            # Validation accuracy chart
            if validation_data and 'validation_results' in validation_data:
                validation_results = validation_data['validation_results']
                accuracies = [result.get('accuracy', 0.0) for result in validation_results]
                
                if accuracies:
                    charts.append({
                        'type': 'line',
                        'title': 'Validation Accuracy Over Time',
                        'data': accuracies,
                        'chart_id': 'validation_accuracy_line'
                    })
            
            return charts
            
        except Exception as e:
            logger.error(f"Error generating charts: {e}")
            return []
    
    async def _generate_recommendations(
        self,
        test_data: Dict[str, Any],
        validation_data: Dict[str, Any] = None,
        comparison_data: Dict[str, Any] = None
    ) -> List[str]:
        """Generate recommendations."""
        try:
            recommendations = []
            
            # Analyze test results
            if 'test_results' in test_data:
                test_results = test_data['test_results']
                total_tests = len(test_results)
                failed_tests = len([r for r in test_results if r.get('status') == 'failed'])
                error_tests = len([r for r in test_results if r.get('status') == 'error'])
                
                if failed_tests > 0:
                    recommendations.append(f"Address {failed_tests} failed tests to improve test suite reliability")
                
                if error_tests > 0:
                    recommendations.append(f"Fix {error_tests} error tests to ensure proper test execution")
                
                # Check execution time
                execution_times = [r.get('execution_time', 0.0) for r in test_results]
                if execution_times:
                    avg_time = np.mean(execution_times)
                    max_time = np.max(execution_times)
                    
                    if avg_time > 10.0:
                        recommendations.append("Consider optimizing test execution time - average time is high")
                    
                    if max_time > 60.0:
                        recommendations.append("Investigate slow tests - some tests take more than 60 seconds")
            
            # Analyze validation results
            if validation_data and 'validation_results' in validation_data:
                validation_results = validation_data['validation_results']
                avg_accuracy = np.mean([r.get('accuracy', 0.0) for r in validation_results])
                
                if avg_accuracy < 0.9:
                    recommendations.append("Improve validation accuracy - current average is below 90%")
                
                failed_validations = len([r for r in validation_results if r.get('status') == 'failed'])
                if failed_validations > 0:
                    recommendations.append(f"Address {failed_validations} failed validations")
            
            # Analyze comparison results
            if comparison_data and 'comparison_results' in comparison_data:
                comparison_results = comparison_data['comparison_results']
                non_matching = len([r for r in comparison_results if not r.get('overall_match', False)])
                
                if non_matching > 0:
                    recommendations.append(f"Investigate {non_matching} non-matching comparisons")
            
            if not recommendations:
                recommendations.append("Test results look good - no immediate issues identified")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"]
    
    async def _generate_html_report(self, config: ReportConfig, report_data: ReportData) -> str:
        """Generate HTML report."""
        try:
            # Create output directory
            output_dir = Path(config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate report file path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{config.name}_{timestamp}.html"
            report_path = output_dir / filename
            
            # Get template
            template = self.report_templates.get('html', self._get_default_html_template())
            
            # Render template
            html_content = template.render(
                report_data=report_data,
                config=config,
                timestamp=datetime.now().isoformat()
            )
            
            # Write file
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            return ""
    
    async def _generate_json_report(self, config: ReportConfig, report_data: ReportData) -> str:
        """Generate JSON report."""
        try:
            # Create output directory
            output_dir = Path(config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate report file path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{config.name}_{timestamp}.json"
            report_path = output_dir / filename
            
            # Convert report data to dictionary
            report_dict = asdict(report_data)
            
            # Add metadata
            report_dict['metadata'] = {
                'generated_at': datetime.now().isoformat(),
                'config': asdict(config),
                'version': '1.0.0'
            }
            
            # Write JSON file
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, indent=2, default=str)
            
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating JSON report: {e}")
            return ""
    
    async def _generate_csv_report(self, config: ReportConfig, report_data: ReportData) -> str:
        """Generate CSV report."""
        try:
            # Create output directory
            output_dir = Path(config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate report file path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{config.name}_{timestamp}.csv"
            report_path = output_dir / filename
            
            # Prepare data for CSV
            csv_data = []
            
            # Add summary data
            summary = report_data.summary
            csv_data.append(['Metric', 'Value'])
            csv_data.append(['Total Tests', summary.get('total_tests', 0)])
            csv_data.append(['Passed Tests', summary.get('passed_tests', 0)])
            csv_data.append(['Failed Tests', summary.get('failed_tests', 0)])
            csv_data.append(['Success Rate', summary.get('success_rate', 0.0)])
            csv_data.append(['Total Execution Time', summary.get('total_execution_time', 0.0)])
            csv_data.append(['Average Execution Time', summary.get('average_execution_time', 0.0)])
            
            # Add test details
            if 'test_results' in report_data.details.get('test_details', {}):
                test_results = report_data.details['test_details']['test_results']
                csv_data.append([])  # Empty row
                csv_data.append(['Test ID', 'Test Name', 'Status', 'Execution Time', 'Error Message'])
                
                for result in test_results:
                    csv_data.append([
                        result.get('test_id', ''),
                        result.get('test_name', ''),
                        result.get('status', ''),
                        result.get('execution_time', 0.0),
                        result.get('error_message', '')
                    ])
            
            # Write CSV file
            with open(report_path, 'w', newline='', encoding='utf-8') as f:
                import csv
                writer = csv.writer(f)
                writer.writerows(csv_data)
            
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating CSV report: {e}")
            return ""
    
    async def _generate_excel_report(self, config: ReportConfig, report_data: ReportData) -> str:
        """Generate Excel report."""
        try:
            # Create output directory
            output_dir = Path(config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate report file path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{config.name}_{timestamp}.xlsx"
            report_path = output_dir / filename
            
            # Create Excel writer
            with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
                # Summary sheet
                summary_df = pd.DataFrame([
                    ['Total Tests', report_data.summary.get('total_tests', 0)],
                    ['Passed Tests', report_data.summary.get('passed_tests', 0)],
                    ['Failed Tests', report_data.summary.get('failed_tests', 0)],
                    ['Success Rate', report_data.summary.get('success_rate', 0.0)],
                    ['Total Execution Time', report_data.summary.get('total_execution_time', 0.0)],
                    ['Average Execution Time', report_data.summary.get('average_execution_time', 0.0)]
                ], columns=['Metric', 'Value'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Test results sheet
                if 'test_results' in report_data.details.get('test_details', {}):
                    test_results = report_data.details['test_details']['test_results']
                    test_df = pd.DataFrame(test_results)
                    test_df.to_excel(writer, sheet_name='Test Results', index=False)
                
                # Recommendations sheet
                recommendations_df = pd.DataFrame(report_data.recommendations, columns=['Recommendation'])
                recommendations_df.to_excel(writer, sheet_name='Recommendations', index=False)
            
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating Excel report: {e}")
            return ""
    
    async def _generate_markdown_report(self, config: ReportConfig, report_data: ReportData) -> str:
        """Generate Markdown report."""
        try:
            # Create output directory
            output_dir = Path(config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate report file path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{config.name}_{timestamp}.md"
            report_path = output_dir / filename
            
            # Generate Markdown content
            markdown_content = self._generate_markdown_content(report_data, config)
            
            # Write file
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating Markdown report: {e}")
            return ""
    
    def _generate_markdown_content(self, report_data: ReportData, config: ReportConfig) -> str:
        """Generate Markdown content."""
        try:
            content = f"""# {config.name}

**Generated:** {report_data.timestamp.strftime('%Y-%m-%d %H:%M:%S')}  
**Report Type:** {report_data.report_type.value}  
**Format:** {config.format.value}

## Summary

| Metric | Value |
|--------|-------|
| Total Tests | {report_data.summary.get('total_tests', 0)} |
| Passed Tests | {report_data.summary.get('passed_tests', 0)} |
| Failed Tests | {report_data.summary.get('failed_tests', 0)} |
| Success Rate | {report_data.summary.get('success_rate', 0.0):.2%} |
| Total Execution Time | {report_data.summary.get('total_execution_time', 0.0):.2f}s |
| Average Execution Time | {report_data.summary.get('average_execution_time', 0.0):.2f}s |

## Test Results

"""
            
            # Add test results
            if 'test_results' in report_data.details.get('test_details', {}):
                test_results = report_data.details['test_details']['test_results']
                content += "| Test ID | Test Name | Status | Execution Time | Error Message |\n"
                content += "|---------|-----------|--------|----------------|---------------|\n"
                
                for result in test_results:
                    content += f"| {result.get('test_id', '')} | {result.get('test_name', '')} | {result.get('status', '')} | {result.get('execution_time', 0.0):.2f}s | {result.get('error_message', '')} |\n"
            
            # Add recommendations
            content += "\n## Recommendations\n\n"
            for i, recommendation in enumerate(report_data.recommendations, 1):
                content += f"{i}. {recommendation}\n"
            
            return content
            
        except Exception as e:
            logger.error(f"Error generating Markdown content: {e}")
            return f"# {config.name}\n\nError generating report: {e}"
    
    def _initialize_default_templates(self):
        """Initialize default report templates."""
        try:
            # HTML template
            html_template = Template("""
<!DOCTYPE html>
<html>
<head>
    <title>{{ config.name }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .summary { margin: 20px 0; }
        .summary table { border-collapse: collapse; width: 100%; }
        .summary th, .summary td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .summary th { background-color: #f2f2f2; }
        .recommendations { margin: 20px 0; }
        .recommendations ul { list-style-type: disc; }
        .footer { margin-top: 40px; font-size: 12px; color: #666; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ config.name }}</h1>
        <p><strong>Generated:</strong> {{ timestamp }}</p>
        <p><strong>Report Type:</strong> {{ report_data.report_type.value }}</p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Tests</td><td>{{ report_data.summary.get('total_tests', 0) }}</td></tr>
            <tr><td>Passed Tests</td><td>{{ report_data.summary.get('passed_tests', 0) }}</td></tr>
            <tr><td>Failed Tests</td><td>{{ report_data.summary.get('failed_tests', 0) }}</td></tr>
            <tr><td>Success Rate</td><td>{{ "%.2f%%"|format(report_data.summary.get('success_rate', 0.0) * 100) }}</td></tr>
            <tr><td>Total Execution Time</td><td>{{ "%.2f"|format(report_data.summary.get('total_execution_time', 0.0)) }}s</td></tr>
            <tr><td>Average Execution Time</td><td>{{ "%.2f"|format(report_data.summary.get('average_execution_time', 0.0)) }}s</td></tr>
        </table>
    </div>
    
    <div class="recommendations">
        <h2>Recommendations</h2>
        <ul>
            {% for recommendation in report_data.recommendations %}
            <li>{{ recommendation }}</li>
            {% endfor %}
        </ul>
    </div>
    
    <div class="footer">
        <p>Report generated by PBF-LB/M Virtual Environment Testing Framework</p>
    </div>
</body>
</html>
            """)
            
            self.report_templates['html'] = html_template
            
        except Exception as e:
            logger.error(f"Error initializing default templates: {e}")
    
    def _get_default_html_template(self) -> Template:
        """Get default HTML template."""
        return Template("""
<!DOCTYPE html>
<html>
<head>
    <title>{{ config.name }}</title>
</head>
<body>
    <h1>{{ config.name }}</h1>
    <p>Generated: {{ timestamp }}</p>
    <p>Report Type: {{ report_data.report_type.value }}</p>
    
    <h2>Summary</h2>
    <p>Total Tests: {{ report_data.summary.get('total_tests', 0) }}</p>
    <p>Passed Tests: {{ report_data.summary.get('passed_tests', 0) }}</p>
    <p>Failed Tests: {{ report_data.summary.get('failed_tests', 0) }}</p>
    <p>Success Rate: {{ "%.2f%%"|format(report_data.summary.get('success_rate', 0.0) * 100) }}</p>
    
    <h2>Recommendations</h2>
    <ul>
        {% for recommendation in report_data.recommendations %}
        <li>{{ recommendation }}</li>
        {% endfor %}
    </ul>
</body>
</html>
        """)


class TestVisualizer:
    """
    Test visualizer for PBF-LB/M virtual environment.
    
    This class provides test visualization capabilities including charts,
    graphs, and interactive visualizations for test results.
    """
    
    def __init__(self):
        """Initialize the test visualizer."""
        self.chart_configs = {}
        self.generated_charts = {}
        
        logger.info("Test Visualizer initialized")
    
    async def create_test_result_chart(
        self,
        test_data: Dict[str, Any],
        chart_type: str = "pie",
        title: str = "Test Results"
    ) -> str:
        """
        Create test result chart.
        
        Args:
            test_data: Test data
            chart_type: Type of chart
            title: Chart title
            
        Returns:
            str: Chart file path
        """
        try:
            # Prepare chart data
            chart_data = self._prepare_chart_data(test_data, chart_type)
            
            # Generate chart
            chart_path = await self._generate_chart(chart_data, chart_type, title)
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Error creating test result chart: {e}")
            return ""
    
    def _prepare_chart_data(self, test_data: Dict[str, Any], chart_type: str) -> Dict[str, Any]:
        """Prepare chart data."""
        try:
            if chart_type == "pie":
                # Count test results by status
                status_counts = {}
                if 'test_results' in test_data:
                    for result in test_data['test_results']:
                        status = result.get('status', 'unknown')
                        status_counts[status] = status_counts.get(status, 0) + 1
                
                return {
                    'labels': list(status_counts.keys()),
                    'values': list(status_counts.values())
                }
            
            elif chart_type == "bar":
                # Execution time by test
                test_names = []
                execution_times = []
                
                if 'test_results' in test_data:
                    for result in test_data['test_results']:
                        test_names.append(result.get('test_name', 'Unknown'))
                        execution_times.append(result.get('execution_time', 0.0))
                
                return {
                    'labels': test_names,
                    'values': execution_times
                }
            
            elif chart_type == "line":
                # Execution time trend
                timestamps = []
                execution_times = []
                
                if 'test_results' in test_data:
                    for result in test_data['test_results']:
                        timestamp = result.get('timestamp', '')
                        if timestamp:
                            timestamps.append(timestamp)
                            execution_times.append(result.get('execution_time', 0.0))
                
                return {
                    'labels': timestamps,
                    'values': execution_times
                }
            
            else:
                return {'labels': [], 'values': []}
                
        except Exception as e:
            logger.error(f"Error preparing chart data: {e}")
            return {'labels': [], 'values': []}
    
    async def _generate_chart(self, chart_data: Dict[str, Any], chart_type: str, title: str) -> str:
        """Generate chart."""
        try:
            # Create output directory
            output_dir = Path("./charts")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate chart file path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{title.replace(' ', '_')}_{timestamp}.png"
            chart_path = output_dir / filename
            
            # Generate chart using matplotlib
            plt.figure(figsize=(10, 6))
            
            if chart_type == "pie":
                plt.pie(chart_data['values'], labels=chart_data['labels'], autopct='%1.1f%%')
            elif chart_type == "bar":
                plt.bar(chart_data['labels'], chart_data['values'])
                plt.xticks(rotation=45)
            elif chart_type == "line":
                plt.plot(chart_data['labels'], chart_data['values'], marker='o')
                plt.xticks(rotation=45)
            
            plt.title(title)
            plt.tight_layout()
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            logger.error(f"Error generating chart: {e}")
            return ""


class TestDocumentation:
    """
    Test documentation generator for PBF-LB/M virtual environment.
    
    This class provides test documentation capabilities including API
    documentation, user guides, and technical documentation.
    """
    
    def __init__(self):
        """Initialize the test documentation generator."""
        self.documentation_templates = {}
        self.generated_docs = {}
        
        logger.info("Test Documentation Generator initialized")
    
    async def generate_api_documentation(
        self,
        api_specs: Dict[str, Any],
        output_directory: str = "./docs"
    ) -> str:
        """
        Generate API documentation.
        
        Args:
            api_specs: API specifications
            output_directory: Output directory
            
        Returns:
            str: Documentation file path
        """
        try:
            # Create output directory
            output_dir = Path(output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate documentation file path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"api_documentation_{timestamp}.md"
            doc_path = output_dir / filename
            
            # Generate documentation content
            doc_content = self._generate_api_doc_content(api_specs)
            
            # Write file
            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write(doc_content)
            
            return str(doc_path)
            
        except Exception as e:
            logger.error(f"Error generating API documentation: {e}")
            return ""
    
    def _generate_api_doc_content(self, api_specs: Dict[str, Any]) -> str:
        """Generate API documentation content."""
        try:
            content = "# API Documentation\n\n"
            content += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # Add API endpoints
            if 'endpoints' in api_specs:
                content += "## API Endpoints\n\n"
                
                for endpoint in api_specs['endpoints']:
                    content += f"### {endpoint.get('method', 'GET')} {endpoint.get('path', '')}\n\n"
                    content += f"**Description:** {endpoint.get('description', '')}\n\n"
                    
                    if 'parameters' in endpoint:
                        content += "**Parameters:**\n\n"
                        for param in endpoint['parameters']:
                            content += f"- `{param.get('name', '')}` ({param.get('type', '')}): {param.get('description', '')}\n"
                        content += "\n"
                    
                    if 'response' in endpoint:
                        content += f"**Response:** {endpoint['response']}\n\n"
            
            return content
            
        except Exception as e:
            logger.error(f"Error generating API documentation content: {e}")
            return f"# API Documentation\n\nError generating documentation: {e}"
