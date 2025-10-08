"""
Quality Dashboard

This module provides quality dashboard data and visualization capabilities for the PBF-LB/M data pipeline.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import statistics

from src.data_pipeline.config.pipeline_config import get_pipeline_config
from src.data_pipeline.quality.monitoring.quality_monitor import QualityDashboard, QualityStatus, QualityMetric
from src.data_pipeline.quality.monitoring.quality_scorer import OverallQualityScore, QualityDimension
from src.data_pipeline.quality.monitoring.trend_analyzer import TrendAnalysis, TrendType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DashboardWidget(Enum):
    """Dashboard widget enumeration."""
    OVERALL_SCORE = "overall_score"
    METRIC_TREND = "metric_trend"
    ANOMALY_ALERT = "anomaly_alert"
    SOURCE_COMPARISON = "source_comparison"
    QUALITY_DISTRIBUTION = "quality_distribution"
    FORECAST = "forecast"
    SEASONAL_PATTERN = "seasonal_pattern"

class ChartType(Enum):
    """Chart type enumeration."""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    TABLE = "table"

@dataclass
class DashboardDataPoint:
    """Dashboard data point data class."""
    timestamp: datetime
    value: float
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DashboardChart:
    """Dashboard chart data class."""
    chart_id: str
    title: str
    chart_type: ChartType
    data: List[DashboardDataPoint] = field(default_factory=list)
    x_axis_label: str = ""
    y_axis_label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DashboardWidget:
    """Dashboard widget data class."""
    widget_id: str
    widget_type: DashboardWidget
    title: str
    charts: List[DashboardChart] = field(default_factory=list)
    position: Tuple[int, int] = (0, 0)
    size: Tuple[int, int] = (1, 1)
    refresh_interval: int = 300  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QualityDashboardData:
    """Quality dashboard data class."""
    dashboard_id: str
    title: str
    source_name: str
    widgets: List[DashboardWidget] = field(default_factory=list)
    overall_metrics: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    refresh_interval: int = 300  # seconds

class QualityDashboardGenerator:
    """
    Quality dashboard data generator for PBF-LB/M data pipeline.
    """
    
    def __init__(self):
        self.config = get_pipeline_config()
        self.dashboard_templates: Dict[str, Dict[str, Any]] = {}
        self.generated_dashboards: Dict[str, QualityDashboardData] = {}
        
        # Initialize dashboard templates
        self._initialize_dashboard_templates()
        
    def generate_quality_dashboard(self, source_name: str, quality_dashboard: QualityDashboard,
                                 quality_score: OverallQualityScore,
                                 trend_analysis: List[TrendAnalysis]) -> QualityDashboardData:
        """
        Generate quality dashboard data for a specific source.
        
        Args:
            source_name: The data source name
            quality_dashboard: The quality dashboard from monitor
            quality_score: The overall quality score
            trend_analysis: List of trend analyses
            
        Returns:
            QualityDashboardData: The generated dashboard data
        """
        try:
            logger.info(f"Generating quality dashboard for {source_name}")
            
            # Create dashboard
            dashboard = QualityDashboardData(
                dashboard_id=f"{source_name}_quality_dashboard",
                title=f"{source_name.replace('_', ' ').title()} Quality Dashboard",
                source_name=source_name
            )
            
            # Generate widgets
            dashboard.widgets = self._generate_dashboard_widgets(
                source_name, quality_dashboard, quality_score, trend_analysis
            )
            
            # Generate overall metrics
            dashboard.overall_metrics = self._generate_overall_metrics(
                quality_dashboard, quality_score, trend_analysis
            )
            
            # Store dashboard
            self.generated_dashboards[source_name] = dashboard
            
            logger.info(f"Generated quality dashboard for {source_name} with {len(dashboard.widgets)} widgets")
            return dashboard
            
        except Exception as e:
            logger.error(f"Error generating quality dashboard for {source_name}: {e}")
            raise
    
    def generate_multi_source_dashboard(self, source_names: List[str]) -> QualityDashboardData:
        """
        Generate a multi-source quality dashboard.
        
        Args:
            source_names: List of source names to include
            
        Returns:
            QualityDashboardData: The generated multi-source dashboard
        """
        try:
            logger.info(f"Generating multi-source dashboard for {len(source_names)} sources")
            
            # Create dashboard
            dashboard = QualityDashboardData(
                dashboard_id="multi_source_quality_dashboard",
                title="Multi-Source Quality Dashboard",
                source_name="multi_source"
            )
            
            # Generate comparison widgets
            dashboard.widgets = self._generate_multi_source_widgets(source_names)
            
            # Generate overall metrics
            dashboard.overall_metrics = self._generate_multi_source_metrics(source_names)
            
            logger.info(f"Generated multi-source dashboard with {len(dashboard.widgets)} widgets")
            return dashboard
            
        except Exception as e:
            logger.error(f"Error generating multi-source dashboard: {e}")
            raise
    
    def get_dashboard_data(self, source_name: str) -> Optional[QualityDashboardData]:
        """
        Get dashboard data for a specific source.
        
        Args:
            source_name: The data source name
            
        Returns:
            QualityDashboardData: The dashboard data, or None if not found
        """
        return self.generated_dashboards.get(source_name)
    
    def get_all_dashboards(self) -> Dict[str, QualityDashboardData]:
        """
        Get all generated dashboards.
        
        Returns:
            Dict[str, QualityDashboardData]: All dashboard data
        """
        return self.generated_dashboards.copy()
    
    def export_dashboard_json(self, source_name: str) -> str:
        """
        Export dashboard data as JSON.
        
        Args:
            source_name: The data source name
            
        Returns:
            str: JSON representation of the dashboard
        """
        try:
            dashboard = self.get_dashboard_data(source_name)
            if not dashboard:
                return json.dumps({"error": "Dashboard not found"})
            
            # Convert to dictionary and handle datetime serialization
            dashboard_dict = asdict(dashboard)
            
            # Convert datetime objects to ISO strings
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetime(item) for item in obj]
                else:
                    return obj
            
            dashboard_dict = convert_datetime(dashboard_dict)
            
            return json.dumps(dashboard_dict, indent=2)
            
        except Exception as e:
            logger.error(f"Error exporting dashboard JSON: {e}")
            return json.dumps({"error": str(e)})
    
    def _generate_dashboard_widgets(self, source_name: str, quality_dashboard: QualityDashboard,
                                  quality_score: OverallQualityScore,
                                  trend_analysis: List[TrendAnalysis]) -> List[DashboardWidget]:
        """Generate widgets for a single source dashboard."""
        try:
            widgets = []
            
            # Overall Score Widget
            widgets.append(self._create_overall_score_widget(source_name, quality_dashboard, quality_score))
            
            # Metric Trend Widget
            widgets.append(self._create_metric_trend_widget(source_name, quality_dashboard, trend_analysis))
            
            # Anomaly Alert Widget
            widgets.append(self._create_anomaly_alert_widget(source_name, quality_dashboard))
            
            # Quality Distribution Widget
            widgets.append(self._create_quality_distribution_widget(source_name, quality_dashboard))
            
            # Forecast Widget (if available)
            forecast_widget = self._create_forecast_widget(source_name, trend_analysis)
            if forecast_widget:
                widgets.append(forecast_widget)
            
            # Seasonal Pattern Widget (if available)
            seasonal_widget = self._create_seasonal_pattern_widget(source_name, trend_analysis)
            if seasonal_widget:
                widgets.append(seasonal_widget)
            
            return widgets
            
        except Exception as e:
            logger.error(f"Error generating dashboard widgets: {e}")
            return []
    
    def _generate_multi_source_widgets(self, source_names: List[str]) -> List[DashboardWidget]:
        """Generate widgets for multi-source dashboard."""
        try:
            widgets = []
            
            # Source Comparison Widget
            widgets.append(self._create_source_comparison_widget(source_names))
            
            # Overall Quality Trend Widget
            widgets.append(self._create_overall_quality_trend_widget(source_names))
            
            # Quality Score Distribution Widget
            widgets.append(self._create_quality_score_distribution_widget(source_names))
            
            return widgets
            
        except Exception as e:
            logger.error(f"Error generating multi-source widgets: {e}")
            return []
    
    def _create_overall_score_widget(self, source_name: str, quality_dashboard: QualityDashboard,
                                   quality_score: OverallQualityScore) -> DashboardWidget:
        """Create overall score widget."""
        try:
            # Create gauge chart
            gauge_chart = DashboardChart(
                chart_id=f"{source_name}_overall_score_gauge",
                title="Overall Quality Score",
                chart_type=ChartType.GAUGE,
                data=[
                    DashboardDataPoint(
                        timestamp=datetime.now(),
                        value=quality_dashboard.overall_score,
                        label="Overall Score",
                        metadata={
                            "min_value": 0.0,
                            "max_value": 1.0,
                            "thresholds": {
                                "excellent": 0.95,
                                "good": 0.90,
                                "fair": 0.80,
                                "poor": 0.70
                            }
                        }
                    )
                ]
            )
            
            # Create dimension scores chart
            dimension_chart = DashboardChart(
                chart_id=f"{source_name}_dimension_scores",
                title="Quality Dimension Scores",
                chart_type=ChartType.BAR,
                data=[
                    DashboardDataPoint(
                        timestamp=datetime.now(),
                        value=score.score,
                        label=score.dimension.value.replace('_', ' ').title(),
                        metadata={"weight": score.weight}
                    )
                    for score in quality_score.dimension_scores
                ],
                y_axis_label="Score",
                metadata={"max_value": 1.0}
            )
            
            return DashboardWidget(
                widget_id=f"{source_name}_overall_score_widget",
                widget_type=DashboardWidget.OVERALL_SCORE,
                title="Overall Quality Score",
                charts=[gauge_chart, dimension_chart],
                position=(0, 0),
                size=(2, 2)
            )
            
        except Exception as e:
            logger.error(f"Error creating overall score widget: {e}")
            return DashboardWidget(
                widget_id=f"{source_name}_overall_score_widget",
                widget_type=DashboardWidget.OVERALL_SCORE,
                title="Overall Quality Score",
                position=(0, 0),
                size=(2, 2)
            )
    
    def _create_metric_trend_widget(self, source_name: str, quality_dashboard: QualityDashboard,
                                  trend_analysis: List[TrendAnalysis]) -> DashboardWidget:
        """Create metric trend widget."""
        try:
            charts = []
            
            # Create trend chart for each metric
            for analysis in trend_analysis:
                if analysis.trend_points:
                    trend_chart = DashboardChart(
                        chart_id=f"{source_name}_{analysis.metric_name}_trend",
                        title=f"{analysis.metric_name.replace('_', ' ').title()} Trend",
                        chart_type=ChartType.LINE,
                        data=[
                            DashboardDataPoint(
                                timestamp=point.timestamp,
                                value=point.value,
                                label="Actual",
                                metadata={"trend_value": point.trend_value, "confidence": point.confidence}
                            )
                            for point in analysis.trend_points[-24:]  # Last 24 points
                        ],
                        x_axis_label="Time",
                        y_axis_label="Score",
                        metadata={
                            "trend_type": analysis.trend_type.value,
                            "trend_strength": analysis.trend_strength.value,
                            "slope": analysis.slope,
                            "r_squared": analysis.r_squared
                        }
                    )
                    charts.append(trend_chart)
            
            return DashboardWidget(
                widget_id=f"{source_name}_metric_trend_widget",
                widget_type=DashboardWidget.METRIC_TREND,
                title="Quality Metric Trends",
                charts=charts,
                position=(2, 0),
                size=(3, 2)
            )
            
        except Exception as e:
            logger.error(f"Error creating metric trend widget: {e}")
            return DashboardWidget(
                widget_id=f"{source_name}_metric_trend_widget",
                widget_type=DashboardWidget.METRIC_TREND,
                title="Quality Metric Trends",
                position=(2, 0),
                size=(3, 2)
            )
    
    def _create_anomaly_alert_widget(self, source_name: str, quality_dashboard: QualityDashboard) -> DashboardWidget:
        """Create anomaly alert widget."""
        try:
            # Create alerts table chart
            alerts_data = []
            for alert in quality_dashboard.alerts:
                alerts_data.append(DashboardDataPoint(
                    timestamp=alert.timestamp,
                    value=alert.current_value,
                    label=alert.metric_name.value,
                    metadata={
                        "threshold": alert.threshold,
                        "severity": alert.severity,
                        "message": alert.message,
                        "resolved": alert.resolved
                    }
                ))
            
            alerts_chart = DashboardChart(
                chart_id=f"{source_name}_anomaly_alerts",
                title="Quality Alerts",
                chart_type=ChartType.TABLE,
                data=alerts_data,
                metadata={
                    "columns": ["Timestamp", "Metric", "Value", "Threshold", "Severity", "Status"],
                    "sort_by": "timestamp",
                    "sort_order": "desc"
                }
            )
            
            # Create alert summary chart
            alert_summary = self._create_alert_summary_chart(source_name, quality_dashboard.alerts)
            
            return DashboardWidget(
                widget_id=f"{source_name}_anomaly_alert_widget",
                widget_type=DashboardWidget.ANOMALY_ALERT,
                title="Quality Alerts & Anomalies",
                charts=[alerts_chart, alert_summary],
                position=(0, 2),
                size=(2, 2)
            )
            
        except Exception as e:
            logger.error(f"Error creating anomaly alert widget: {e}")
            return DashboardWidget(
                widget_id=f"{source_name}_anomaly_alert_widget",
                widget_type=DashboardWidget.ANOMALY_ALERT,
                title="Quality Alerts & Anomalies",
                position=(0, 2),
                size=(2, 2)
            )
    
    def _create_alert_summary_chart(self, source_name: str, alerts: List[Any]) -> DashboardChart:
        """Create alert summary chart."""
        try:
            # Count alerts by severity
            severity_counts = {}
            for alert in alerts:
                severity = alert.severity
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            summary_data = [
                DashboardDataPoint(
                    timestamp=datetime.now(),
                    value=count,
                    label=severity.title(),
                    metadata={"severity": severity}
                )
                for severity, count in severity_counts.items()
            ]
            
            return DashboardChart(
                chart_id=f"{source_name}_alert_summary",
                title="Alert Summary",
                chart_type=ChartType.PIE,
                data=summary_data
            )
            
        except Exception as e:
            logger.error(f"Error creating alert summary chart: {e}")
            return DashboardChart(
                chart_id=f"{source_name}_alert_summary",
                title="Alert Summary",
                chart_type=ChartType.PIE
            )
    
    def _create_quality_distribution_widget(self, source_name: str, quality_dashboard: QualityDashboard) -> DashboardWidget:
        """Create quality distribution widget."""
        try:
            # Create metrics distribution chart
            metrics_data = [
                DashboardDataPoint(
                    timestamp=datetime.now(),
                    value=metric.value,
                    label=metric.metric_name.value.replace('_', ' ').title(),
                    metadata={
                        "threshold": metric.threshold,
                        "status": metric.status.value
                    }
                )
                for metric in quality_dashboard.metrics
            ]
            
            distribution_chart = DashboardChart(
                chart_id=f"{source_name}_quality_distribution",
                title="Quality Metrics Distribution",
                chart_type=ChartType.BAR,
                data=metrics_data,
                y_axis_label="Score",
                metadata={"max_value": 1.0}
            )
            
            return DashboardWidget(
                widget_id=f"{source_name}_quality_distribution_widget",
                widget_type=DashboardWidget.QUALITY_DISTRIBUTION,
                title="Quality Distribution",
                charts=[distribution_chart],
                position=(2, 2),
                size=(2, 2)
            )
            
        except Exception as e:
            logger.error(f"Error creating quality distribution widget: {e}")
            return DashboardWidget(
                widget_id=f"{source_name}_quality_distribution_widget",
                widget_type=DashboardWidget.QUALITY_DISTRIBUTION,
                title="Quality Distribution",
                position=(2, 2),
                size=(2, 2)
            )
    
    def _create_forecast_widget(self, source_name: str, trend_analysis: List[TrendAnalysis]) -> Optional[DashboardWidget]:
        """Create forecast widget if forecast data is available."""
        try:
            forecast_charts = []
            
            for analysis in trend_analysis:
                if analysis.forecast:
                    forecast_data = [
                        DashboardDataPoint(
                            timestamp=forecast_point["timestamp"],
                            value=forecast_point["value"],
                            label="Forecast",
                            metadata={
                                "lower_bound": forecast_point.get("lower_bound", 0.0),
                                "upper_bound": forecast_point.get("upper_bound", 0.0),
                                "confidence": forecast_point.get("confidence", 0.95)
                            }
                        )
                        for forecast_point in analysis.forecast
                    ]
                    
                    forecast_chart = DashboardChart(
                        chart_id=f"{source_name}_{analysis.metric_name}_forecast",
                        title=f"{analysis.metric_name.replace('_', ' ').title()} Forecast",
                        chart_type=ChartType.LINE,
                        data=forecast_data,
                        x_axis_label="Time",
                        y_axis_label="Score",
                        metadata={"forecast_horizon": len(analysis.forecast)}
                    )
                    forecast_charts.append(forecast_chart)
            
            if forecast_charts:
                return DashboardWidget(
                    widget_id=f"{source_name}_forecast_widget",
                    widget_type=DashboardWidget.FORECAST,
                    title="Quality Forecast",
                    charts=forecast_charts,
                    position=(4, 0),
                    size=(3, 2)
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating forecast widget: {e}")
            return None
    
    def _create_seasonal_pattern_widget(self, source_name: str, trend_analysis: List[TrendAnalysis]) -> Optional[DashboardWidget]:
        """Create seasonal pattern widget if seasonal data is available."""
        try:
            seasonal_charts = []
            
            for analysis in trend_analysis:
                if analysis.seasonal_patterns:
                    # Daily pattern chart
                    if "daily" in analysis.seasonal_patterns:
                        daily_pattern = analysis.seasonal_patterns["daily"]
                        daily_data = [
                            DashboardDataPoint(
                                timestamp=datetime.now().replace(hour=hour, minute=0, second=0, microsecond=0),
                                value=value,
                                label=f"Hour {hour}",
                                metadata={"hour": hour}
                            )
                            for hour, value in enumerate(daily_pattern["hourly_means"])
                        ]
                        
                        daily_chart = DashboardChart(
                            chart_id=f"{source_name}_{analysis.metric_name}_daily_pattern",
                            title=f"{analysis.metric_name.replace('_', ' ').title()} Daily Pattern",
                            chart_type=ChartType.LINE,
                            data=daily_data,
                            x_axis_label="Hour",
                            y_axis_label="Score",
                            metadata={
                                "peak_hour": daily_pattern.get("peak_hour", 0),
                                "valley_hour": daily_pattern.get("valley_hour", 0),
                                "amplitude": daily_pattern.get("amplitude", 0.0)
                            }
                        )
                        seasonal_charts.append(daily_chart)
                    
                    # Weekly pattern chart
                    if "weekly" in analysis.seasonal_patterns:
                        weekly_pattern = analysis.seasonal_patterns["weekly"]
                        weekly_data = [
                            DashboardDataPoint(
                                timestamp=datetime.now(),
                                value=value,
                                label=f"Day {day}",
                                metadata={"day": day}
                            )
                            for day, value in enumerate(weekly_pattern["daily_means"])
                        ]
                        
                        weekly_chart = DashboardChart(
                            chart_id=f"{source_name}_{analysis.metric_name}_weekly_pattern",
                            title=f"{analysis.metric_name.replace('_', ' ').title()} Weekly Pattern",
                            chart_type=ChartType.BAR,
                            data=weekly_data,
                            x_axis_label="Day of Week",
                            y_axis_label="Score",
                            metadata={
                                "peak_day": weekly_pattern.get("peak_day", 0),
                                "valley_day": weekly_pattern.get("valley_day", 0),
                                "amplitude": weekly_pattern.get("amplitude", 0.0)
                            }
                        )
                        seasonal_charts.append(weekly_chart)
            
            if seasonal_charts:
                return DashboardWidget(
                    widget_id=f"{source_name}_seasonal_pattern_widget",
                    widget_type=DashboardWidget.SEASONAL_PATTERN,
                    title="Seasonal Patterns",
                    charts=seasonal_charts,
                    position=(4, 2),
                    size=(3, 2)
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating seasonal pattern widget: {e}")
            return None
    
    def _create_source_comparison_widget(self, source_names: List[str]) -> DashboardWidget:
        """Create source comparison widget."""
        try:
            # Get latest quality scores for each source
            comparison_data = []
            for source_name in source_names:
                dashboard = self.get_dashboard_data(source_name)
                if dashboard and dashboard.overall_metrics:
                    overall_score = dashboard.overall_metrics.get("overall_score", 0.0)
                    comparison_data.append(DashboardDataPoint(
                        timestamp=datetime.now(),
                        value=overall_score,
                        label=source_name.replace('_', ' ').title(),
                        metadata={"source": source_name}
                    ))
            
            comparison_chart = DashboardChart(
                chart_id="source_comparison",
                title="Source Quality Comparison",
                chart_type=ChartType.BAR,
                data=comparison_data,
                y_axis_label="Overall Score",
                metadata={"max_value": 1.0}
            )
            
            return DashboardWidget(
                widget_id="source_comparison_widget",
                widget_type=DashboardWidget.SOURCE_COMPARISON,
                title="Source Comparison",
                charts=[comparison_chart],
                position=(0, 0),
                size=(3, 2)
            )
            
        except Exception as e:
            logger.error(f"Error creating source comparison widget: {e}")
            return DashboardWidget(
                widget_id="source_comparison_widget",
                widget_type=DashboardWidget.SOURCE_COMPARISON,
                title="Source Comparison",
                position=(0, 0),
                size=(3, 2)
            )
    
    def _create_overall_quality_trend_widget(self, source_names: List[str]) -> DashboardWidget:
        """Create overall quality trend widget."""
        try:
            trend_charts = []
            
            for source_name in source_names:
                dashboard = self.get_dashboard_data(source_name)
                if dashboard and dashboard.overall_metrics:
                    # Create trend data (simplified - would need historical data)
                    trend_data = [
                        DashboardDataPoint(
                            timestamp=datetime.now() - timedelta(hours=i),
                            value=dashboard.overall_metrics.get("overall_score", 0.0),
                            label=source_name.replace('_', ' ').title(),
                            metadata={"source": source_name}
                        )
                        for i in range(24, 0, -1)
                    ]
                    
                    trend_chart = DashboardChart(
                        chart_id=f"{source_name}_overall_trend",
                        title=f"{source_name.replace('_', ' ').title()} Overall Trend",
                        chart_type=ChartType.LINE,
                        data=trend_data,
                        x_axis_label="Time",
                        y_axis_label="Overall Score",
                        metadata={"source": source_name}
                    )
                    trend_charts.append(trend_chart)
            
            return DashboardWidget(
                widget_id="overall_quality_trend_widget",
                widget_type=DashboardWidget.METRIC_TREND,
                title="Overall Quality Trends",
                charts=trend_charts,
                position=(3, 0),
                size=(3, 2)
            )
            
        except Exception as e:
            logger.error(f"Error creating overall quality trend widget: {e}")
            return DashboardWidget(
                widget_id="overall_quality_trend_widget",
                widget_type=DashboardWidget.METRIC_TREND,
                title="Overall Quality Trends",
                position=(3, 0),
                size=(3, 2)
            )
    
    def _create_quality_score_distribution_widget(self, source_names: List[str]) -> DashboardWidget:
        """Create quality score distribution widget."""
        try:
            # Collect all quality scores
            all_scores = []
            for source_name in source_names:
                dashboard = self.get_dashboard_data(source_name)
                if dashboard and dashboard.overall_metrics:
                    overall_score = dashboard.overall_metrics.get("overall_score", 0.0)
                    all_scores.append(overall_score)
            
            if not all_scores:
                return DashboardWidget(
                    widget_id="quality_score_distribution_widget",
                    widget_type=DashboardWidget.QUALITY_DISTRIBUTION,
                    title="Quality Score Distribution",
                    position=(0, 2),
                    size=(3, 2)
                )
            
            # Create distribution data
            distribution_data = [
                DashboardDataPoint(
                    timestamp=datetime.now(),
                    value=count,
                    label=f"{score_range[0]:.1f}-{score_range[1]:.1f}",
                    metadata={"range": score_range}
                )
                for score_range, count in self._create_score_bins(all_scores).items()
            ]
            
            distribution_chart = DashboardChart(
                chart_id="quality_score_distribution",
                title="Quality Score Distribution",
                chart_type=ChartType.BAR,
                data=distribution_data,
                x_axis_label="Score Range",
                y_axis_label="Count"
            )
            
            return DashboardWidget(
                widget_id="quality_score_distribution_widget",
                widget_type=DashboardWidget.QUALITY_DISTRIBUTION,
                title="Quality Score Distribution",
                charts=[distribution_chart],
                position=(0, 2),
                size=(3, 2)
            )
            
        except Exception as e:
            logger.error(f"Error creating quality score distribution widget: {e}")
            return DashboardWidget(
                widget_id="quality_score_distribution_widget",
                widget_type=DashboardWidget.QUALITY_DISTRIBUTION,
                title="Quality Score Distribution",
                position=(0, 2),
                size=(3, 2)
            )
    
    def _create_score_bins(self, scores: List[float]) -> Dict[Tuple[float, float], int]:
        """Create score bins for distribution."""
        try:
            if not scores:
                return {}
            
            # Create bins
            bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
            bin_counts = {bin_range: 0 for bin_range in bins}
            
            for score in scores:
                for bin_range in bins:
                    if bin_range[0] <= score < bin_range[1]:
                        bin_counts[bin_range] += 1
                        break
                else:
                    # Handle edge case for score = 1.0
                    if score == 1.0:
                        bin_counts[(0.8, 1.0)] += 1
            
            return bin_counts
            
        except Exception as e:
            logger.error(f"Error creating score bins: {e}")
            return {}
    
    def _generate_overall_metrics(self, quality_dashboard: QualityDashboard,
                                quality_score: OverallQualityScore,
                                trend_analysis: List[TrendAnalysis]) -> Dict[str, Any]:
        """Generate overall metrics for the dashboard."""
        try:
            metrics = {
                "overall_score": quality_dashboard.overall_score,
                "status": quality_dashboard.status.value,
                "total_metrics": len(quality_dashboard.metrics),
                "active_alerts": len([alert for alert in quality_dashboard.alerts if not alert.resolved]),
                "total_alerts": len(quality_dashboard.alerts),
                "confidence_level": quality_score.confidence_level,
                "scoring_method": quality_score.scoring_method.value,
                "trend_analyses_count": len(trend_analysis),
                "last_updated": datetime.now().isoformat()
            }
            
            # Add trend summary
            if trend_analysis:
                trend_types = [analysis.trend_type.value for analysis in trend_analysis]
                metrics["trend_summary"] = {
                    "dominant_trend": max(set(trend_types), key=trend_types.count),
                    "trend_types": trend_types,
                    "average_r_squared": statistics.mean([analysis.r_squared for analysis in trend_analysis])
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error generating overall metrics: {e}")
            return {}
    
    def _generate_multi_source_metrics(self, source_names: List[str]) -> Dict[str, Any]:
        """Generate overall metrics for multi-source dashboard."""
        try:
            metrics = {
                "total_sources": len(source_names),
                "sources": {},
                "overall_summary": {},
                "last_updated": datetime.now().isoformat()
            }
            
            # Collect metrics for each source
            all_scores = []
            for source_name in source_names:
                dashboard = self.get_dashboard_data(source_name)
                if dashboard and dashboard.overall_metrics:
                    source_metrics = dashboard.overall_metrics
                    metrics["sources"][source_name] = source_metrics
                    all_scores.append(source_metrics.get("overall_score", 0.0))
            
            # Generate overall summary
            if all_scores:
                metrics["overall_summary"] = {
                    "average_score": statistics.mean(all_scores),
                    "median_score": statistics.median(all_scores),
                    "min_score": min(all_scores),
                    "max_score": max(all_scores),
                    "score_range": max(all_scores) - min(all_scores),
                    "standard_deviation": statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error generating multi-source metrics: {e}")
            return {}
    
    def _initialize_dashboard_templates(self):
        """Initialize dashboard templates."""
        try:
            # PBF Process template
            self.dashboard_templates["pbf_process"] = {
                "widgets": [
                    {"type": "overall_score", "position": (0, 0), "size": (2, 2)},
                    {"type": "metric_trend", "position": (2, 0), "size": (3, 2)},
                    {"type": "anomaly_alert", "position": (0, 2), "size": (2, 2)},
                    {"type": "quality_distribution", "position": (2, 2), "size": (2, 2)},
                    {"type": "forecast", "position": (4, 0), "size": (3, 2)},
                    {"type": "seasonal_pattern", "position": (4, 2), "size": (3, 2)}
                ],
                "refresh_interval": 300
            }
            
            # ISPM Monitoring template
            self.dashboard_templates["ispm_monitoring"] = {
                "widgets": [
                    {"type": "overall_score", "position": (0, 0), "size": (2, 2)},
                    {"type": "metric_trend", "position": (2, 0), "size": (3, 2)},
                    {"type": "anomaly_alert", "position": (0, 2), "size": (2, 2)},
                    {"type": "quality_distribution", "position": (2, 2), "size": (2, 2)}
                ],
                "refresh_interval": 60  # More frequent for real-time monitoring
            }
            
            # CT Scan template
            self.dashboard_templates["ct_scan"] = {
                "widgets": [
                    {"type": "overall_score", "position": (0, 0), "size": (2, 2)},
                    {"type": "metric_trend", "position": (2, 0), "size": (3, 2)},
                    {"type": "anomaly_alert", "position": (0, 2), "size": (2, 2)},
                    {"type": "quality_distribution", "position": (2, 2), "size": (2, 2)}
                ],
                "refresh_interval": 600  # Less frequent for batch processing
            }
            
            # Powder Bed template
            self.dashboard_templates["powder_bed"] = {
                "widgets": [
                    {"type": "overall_score", "position": (0, 0), "size": (2, 2)},
                    {"type": "metric_trend", "position": (2, 0), "size": (3, 2)},
                    {"type": "anomaly_alert", "position": (0, 2), "size": (2, 2)},
                    {"type": "quality_distribution", "position": (2, 2), "size": (2, 2)}
                ],
                "refresh_interval": 300
            }
            
            logger.info("Initialized dashboard templates")
            
        except Exception as e:
            logger.error(f"Error initializing dashboard templates: {e}")
