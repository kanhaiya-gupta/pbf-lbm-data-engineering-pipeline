"""
Trend Analyzer

This module provides quality trend analysis capabilities for the PBF-LB/M data pipeline.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import statistics
from scipy import stats
from scipy.signal import find_peaks
import warnings

from src.data_pipeline.config.pipeline_config import get_pipeline_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress scipy warnings
warnings.filterwarnings('ignore', category=UserWarning)

class TrendType(Enum):
    """Trend type enumeration."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    CYCLICAL = "cyclical"
    SEASONAL = "seasonal"
    IRREGULAR = "irregular"

class TrendStrength(Enum):
    """Trend strength enumeration."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

class AnomalyType(Enum):
    """Anomaly type enumeration."""
    SPIKE = "spike"
    DROP = "drop"
    OUTLIER = "outlier"
    PATTERN_BREAK = "pattern_break"
    SEASONAL_ANOMALY = "seasonal_anomaly"

@dataclass
class TrendPoint:
    """Trend point data class."""
    timestamp: datetime
    value: float
    trend_value: float
    confidence: float

@dataclass
class TrendAnalysis:
    """Trend analysis data class."""
    source_name: str
    metric_name: str
    trend_type: TrendType
    trend_strength: TrendStrength
    slope: float
    r_squared: float
    p_value: float
    confidence_interval: Tuple[float, float]
    trend_points: List[TrendPoint] = field(default_factory=list)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    seasonal_patterns: Dict[str, Any] = field(default_factory=dict)
    forecast: List[Dict[str, Any]] = field(default_factory=list)
    analysis_timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class QualityTrendReport:
    """Quality trend report data class."""
    report_id: str
    source_name: str
    period_start: datetime
    period_end: datetime
    overall_trend: TrendType
    trend_analyses: List[TrendAnalysis] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)

class TrendAnalyzer:
    """
    Quality trend analysis service for PBF-LB/M data pipeline.
    """
    
    def __init__(self):
        self.config = get_pipeline_config()
        self.trend_analyses: Dict[str, List[TrendAnalysis]] = {}
        self.quality_data: Dict[str, List[Dict[str, Any]]] = {}
        
        # Analysis parameters
        self.min_data_points = 10
        self.anomaly_threshold = 2.0  # Standard deviations
        self.seasonal_periods = [24, 168]  # Daily and weekly patterns
        self.forecast_horizon = 24  # Hours
        
    def analyze_quality_trends(self, source_name: str, quality_data: List[Dict[str, Any]], 
                             metric_name: str = "overall_score") -> TrendAnalysis:
        """
        Analyze quality trends for a specific source and metric.
        
        Args:
            source_name: The data source name
            quality_data: List of quality data points
            metric_name: The metric to analyze
            
        Returns:
            TrendAnalysis: The trend analysis results
        """
        try:
            logger.info(f"Analyzing quality trends for {source_name} - {metric_name}")
            
            # Prepare data
            df = self._prepare_trend_data(quality_data, metric_name)
            
            if len(df) < self.min_data_points:
                return self._create_insufficient_data_analysis(source_name, metric_name)
            
            # Perform trend analysis
            trend_analysis = self._perform_trend_analysis(source_name, metric_name, df)
            
            # Detect anomalies
            trend_analysis.anomalies = self._detect_anomalies(df, metric_name)
            
            # Analyze seasonal patterns
            trend_analysis.seasonal_patterns = self._analyze_seasonal_patterns(df, metric_name)
            
            # Generate forecast
            trend_analysis.forecast = self._generate_forecast(df, metric_name)
            
            # Store analysis
            if source_name not in self.trend_analyses:
                self.trend_analyses[source_name] = []
            self.trend_analyses[source_name].append(trend_analysis)
            
            logger.info(f"Trend analysis completed for {source_name} - {metric_name}. Trend: {trend_analysis.trend_type.value}")
            return trend_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing quality trends for {source_name}: {e}")
            raise
    
    def analyze_pbf_process_trends(self, quality_data: List[Dict[str, Any]]) -> List[TrendAnalysis]:
        """
        Analyze quality trends for PBF process data.
        
        Args:
            quality_data: List of PBF process quality data
            
        Returns:
            List[TrendAnalysis]: List of trend analyses for different metrics
        """
        try:
            logger.info(f"Analyzing PBF process quality trends for {len(quality_data)} data points")
            
            analyses = []
            metrics = ["overall_score", "completeness", "accuracy", "consistency", "timeliness", "validity"]
            
            for metric in metrics:
                analysis = self.analyze_quality_trends("pbf_process", quality_data, metric)
                analyses.append(analysis)
            
            return analyses
            
        except Exception as e:
            logger.error(f"Error analyzing PBF process trends: {e}")
            raise
    
    def analyze_ispm_monitoring_trends(self, quality_data: List[Dict[str, Any]]) -> List[TrendAnalysis]:
        """
        Analyze quality trends for ISPM monitoring data.
        
        Args:
            quality_data: List of ISPM monitoring quality data
            
        Returns:
            List[TrendAnalysis]: List of trend analyses for different metrics
        """
        try:
            logger.info(f"Analyzing ISPM monitoring quality trends for {len(quality_data)} data points")
            
            analyses = []
            metrics = ["overall_score", "completeness", "accuracy", "consistency", "timeliness", "validity"]
            
            for metric in metrics:
                analysis = self.analyze_quality_trends("ispm_monitoring", quality_data, metric)
                analyses.append(analysis)
            
            return analyses
            
        except Exception as e:
            logger.error(f"Error analyzing ISPM monitoring trends: {e}")
            raise
    
    def analyze_ct_scan_trends(self, quality_data: List[Dict[str, Any]]) -> List[TrendAnalysis]:
        """
        Analyze quality trends for CT scan data.
        
        Args:
            quality_data: List of CT scan quality data
            
        Returns:
            List[TrendAnalysis]: List of trend analyses for different metrics
        """
        try:
            logger.info(f"Analyzing CT scan quality trends for {len(quality_data)} data points")
            
            analyses = []
            metrics = ["overall_score", "completeness", "accuracy", "consistency", "timeliness", "validity"]
            
            for metric in metrics:
                analysis = self.analyze_quality_trends("ct_scan", quality_data, metric)
                analyses.append(analysis)
            
            return analyses
            
        except Exception as e:
            logger.error(f"Error analyzing CT scan trends: {e}")
            raise
    
    def analyze_powder_bed_trends(self, quality_data: List[Dict[str, Any]]) -> List[TrendAnalysis]:
        """
        Analyze quality trends for powder bed data.
        
        Args:
            quality_data: List of powder bed quality data
            
        Returns:
            List[TrendAnalysis]: List of trend analyses for different metrics
        """
        try:
            logger.info(f"Analyzing powder bed quality trends for {len(quality_data)} data points")
            
            analyses = []
            metrics = ["overall_score", "completeness", "accuracy", "consistency", "timeliness", "validity"]
            
            for metric in metrics:
                analysis = self.analyze_quality_trends("powder_bed", quality_data, metric)
                analyses.append(analysis)
            
            return analyses
            
        except Exception as e:
            logger.error(f"Error analyzing powder bed trends: {e}")
            raise
    
    def generate_trend_report(self, source_name: str, period_hours: int = 168) -> QualityTrendReport:
        """
        Generate a comprehensive trend report for a data source.
        
        Args:
            source_name: The data source name
            period_hours: Number of hours to include in the report
            
        Returns:
            QualityTrendReport: The generated trend report
        """
        try:
            logger.info(f"Generating trend report for {source_name} (last {period_hours} hours)")
            
            period_end = datetime.now()
            period_start = period_end - timedelta(hours=period_hours)
            
            # Get trend analyses for the period
            if source_name not in self.trend_analyses:
                return self._create_no_data_report(source_name, period_start, period_end)
            
            recent_analyses = [
                analysis for analysis in self.trend_analyses[source_name]
                if analysis.analysis_timestamp >= period_start
            ]
            
            if not recent_analyses:
                return self._create_no_data_report(source_name, period_start, period_end)
            
            # Determine overall trend
            overall_trend = self._determine_overall_trend(recent_analyses)
            
            # Generate summary
            summary = self._generate_trend_summary(recent_analyses)
            
            # Generate recommendations
            recommendations = self._generate_trend_recommendations(recent_analyses, overall_trend)
            
            # Create report
            report = QualityTrendReport(
                report_id=f"{source_name}_trend_{int(datetime.now().timestamp())}",
                source_name=source_name,
                period_start=period_start,
                period_end=period_end,
                overall_trend=overall_trend,
                trend_analyses=recent_analyses,
                summary=summary,
                recommendations=recommendations
            )
            
            logger.info(f"Generated trend report for {source_name}. Overall trend: {overall_trend.value}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating trend report for {source_name}: {e}")
            raise
    
    def compare_trends(self, source_names: List[str]) -> Dict[str, Any]:
        """
        Compare trends across multiple data sources.
        
        Args:
            source_names: List of source names to compare
            
        Returns:
            Dict[str, Any]: Comparison results
        """
        try:
            logger.info(f"Comparing trends across {len(source_names)} sources")
            
            comparison = {
                "sources": {},
                "trend_comparison": {},
                "summary": {}
            }
            
            # Get latest analyses for each source
            for source_name in source_names:
                if source_name in self.trend_analyses and self.trend_analyses[source_name]:
                    latest_analysis = self.trend_analyses[source_name][-1]
                    comparison["sources"][source_name] = {
                        "trend_type": latest_analysis.trend_type.value,
                        "trend_strength": latest_analysis.trend_strength.value,
                        "slope": latest_analysis.slope,
                        "r_squared": latest_analysis.r_squared,
                        "confidence": latest_analysis.confidence_interval,
                        "anomalies_count": len(latest_analysis.anomalies)
                    }
            
            # Compare trends
            comparison["trend_comparison"] = self._compare_trend_types(comparison["sources"])
            
            # Generate summary
            comparison["summary"] = self._generate_comparison_summary(comparison["sources"])
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing trends: {e}")
            return {"error": str(e)}
    
    def get_trend_forecast(self, source_name: str, metric_name: str, hours_ahead: int = 24) -> List[Dict[str, Any]]:
        """
        Get trend forecast for a specific source and metric.
        
        Args:
            source_name: The data source name
            metric_name: The metric name
            hours_ahead: Number of hours to forecast
            
        Returns:
            List[Dict[str, Any]]: Forecast data points
        """
        try:
            if source_name not in self.trend_analyses:
                return []
            
            # Find the latest analysis for the metric
            latest_analysis = None
            for analysis in reversed(self.trend_analyses[source_name]):
                if analysis.metric_name == metric_name:
                    latest_analysis = analysis
                    break
            
            if not latest_analysis or not latest_analysis.forecast:
                return []
            
            # Return forecast for the requested period
            return latest_analysis.forecast[:hours_ahead]
            
        except Exception as e:
            logger.error(f"Error getting trend forecast: {e}")
            return []
    
    def _prepare_trend_data(self, quality_data: List[Dict[str, Any]], metric_name: str) -> pd.DataFrame:
        """Prepare data for trend analysis."""
        try:
            if not quality_data:
                return pd.DataFrame()
            
            # Extract relevant data
            data_points = []
            for record in quality_data:
                if metric_name in record and 'timestamp' in record:
                    try:
                        timestamp = pd.to_datetime(record['timestamp'])
                        value = float(record[metric_name])
                        data_points.append({
                            'timestamp': timestamp,
                            'value': value
                        })
                    except (ValueError, TypeError):
                        continue
            
            if not data_points:
                return pd.DataFrame()
            
            # Create DataFrame and sort by timestamp
            df = pd.DataFrame(data_points)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Remove duplicates and interpolate missing values
            df = df.drop_duplicates(subset=['timestamp'])
            df = df.set_index('timestamp').resample('1H').mean().interpolate().reset_index()
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing trend data: {e}")
            return pd.DataFrame()
    
    def _perform_trend_analysis(self, source_name: str, metric_name: str, df: pd.DataFrame) -> TrendAnalysis:
        """Perform trend analysis on the data."""
        try:
            if df.empty:
                return self._create_insufficient_data_analysis(source_name, metric_name)
            
            # Prepare data for analysis
            x = np.arange(len(df))
            y = df['value'].values
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            r_squared = r_value ** 2
            
            # Determine trend type and strength
            trend_type = self._determine_trend_type(slope, p_value, df)
            trend_strength = self._determine_trend_strength(r_squared)
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(slope, std_err)
            
            # Create trend points
            trend_points = []
            for i, (timestamp, value) in enumerate(df.iterrows()):
                trend_value = slope * i + intercept
                confidence = max(0.0, min(1.0, r_squared))
                trend_points.append(TrendPoint(
                    timestamp=timestamp,
                    value=value,
                    trend_value=trend_value,
                    confidence=confidence
                ))
            
            return TrendAnalysis(
                source_name=source_name,
                metric_name=metric_name,
                trend_type=trend_type,
                trend_strength=trend_strength,
                slope=slope,
                r_squared=r_squared,
                p_value=p_value,
                confidence_interval=confidence_interval,
                trend_points=trend_points
            )
            
        except Exception as e:
            logger.error(f"Error performing trend analysis: {e}")
            return self._create_insufficient_data_analysis(source_name, metric_name)
    
    def _determine_trend_type(self, slope: float, p_value: float, df: pd.DataFrame) -> TrendType:
        """Determine the type of trend."""
        try:
            # Check for cyclical patterns
            if self._has_cyclical_pattern(df):
                return TrendType.CYCLICAL
            
            # Check for seasonal patterns
            if self._has_seasonal_pattern(df):
                return TrendType.SEASONAL
            
            # Determine linear trend
            if p_value < 0.05:  # Statistically significant
                if slope > 0.01:
                    return TrendType.INCREASING
                elif slope < -0.01:
                    return TrendType.DECREASING
                else:
                    return TrendType.STABLE
            else:
                return TrendType.STABLE
                
        except Exception as e:
            logger.error(f"Error determining trend type: {e}")
            return TrendType.IRREGULAR
    
    def _determine_trend_strength(self, r_squared: float) -> TrendStrength:
        """Determine the strength of the trend."""
        try:
            if r_squared >= 0.8:
                return TrendStrength.VERY_STRONG
            elif r_squared >= 0.6:
                return TrendStrength.STRONG
            elif r_squared >= 0.4:
                return TrendStrength.MODERATE
            else:
                return TrendStrength.WEAK
                
        except Exception as e:
            logger.error(f"Error determining trend strength: {e}")
            return TrendStrength.WEAK
    
    def _calculate_confidence_interval(self, slope: float, std_err: float, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for the slope."""
        try:
            # 95% confidence interval
            alpha = 1 - confidence
            t_value = stats.t.ppf(1 - alpha/2, df=100)  # Approximate degrees of freedom
            
            margin_error = t_value * std_err
            lower_bound = slope - margin_error
            upper_bound = slope + margin_error
            
            return (lower_bound, upper_bound)
            
        except Exception as e:
            logger.error(f"Error calculating confidence interval: {e}")
            return (0.0, 0.0)
    
    def _detect_anomalies(self, df: pd.DataFrame, metric_name: str) -> List[Dict[str, Any]]:
        """Detect anomalies in the data."""
        try:
            if df.empty or len(df) < 3:
                return []
            
            anomalies = []
            values = df['value'].values
            
            # Statistical anomaly detection
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            for i, (timestamp, value) in enumerate(df.iterrows()):
                z_score = abs(value - mean_val) / std_val if std_val > 0 else 0
                
                if z_score > self.anomaly_threshold:
                    anomaly_type = AnomalyType.SPIKE if value > mean_val else AnomalyType.DROP
                    anomalies.append({
                        "timestamp": timestamp,
                        "value": value,
                        "z_score": z_score,
                        "type": anomaly_type.value,
                        "severity": "high" if z_score > 3.0 else "medium"
                    })
            
            # Pattern break detection
            pattern_breaks = self._detect_pattern_breaks(df)
            anomalies.extend(pattern_breaks)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []
    
    def _detect_pattern_breaks(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect pattern breaks in the data."""
        try:
            if len(df) < 10:
                return []
            
            pattern_breaks = []
            values = df['value'].values
            
            # Use change point detection (simplified version)
            window_size = min(5, len(values) // 3)
            
            for i in range(window_size, len(values) - window_size):
                before_mean = np.mean(values[i-window_size:i])
                after_mean = np.mean(values[i:i+window_size])
                
                # Check for significant change
                if abs(after_mean - before_mean) > 2 * np.std(values):
                    pattern_breaks.append({
                        "timestamp": df.iloc[i]['timestamp'],
                        "value": values[i],
                        "type": AnomalyType.PATTERN_BREAK.value,
                        "before_mean": before_mean,
                        "after_mean": after_mean,
                        "change_magnitude": abs(after_mean - before_mean)
                    })
            
            return pattern_breaks
            
        except Exception as e:
            logger.error(f"Error detecting pattern breaks: {e}")
            return []
    
    def _analyze_seasonal_patterns(self, df: pd.DataFrame, metric_name: str) -> Dict[str, Any]:
        """Analyze seasonal patterns in the data."""
        try:
            if df.empty or len(df) < 24:  # Need at least 24 hours of data
                return {}
            
            seasonal_patterns = {}
            values = df['value'].values
            
            # Daily pattern (24-hour cycle)
            if len(values) >= 24:
                daily_pattern = self._analyze_daily_pattern(values)
                if daily_pattern:
                    seasonal_patterns["daily"] = daily_pattern
            
            # Weekly pattern (168-hour cycle)
            if len(values) >= 168:
                weekly_pattern = self._analyze_weekly_pattern(values)
                if weekly_pattern:
                    seasonal_patterns["weekly"] = weekly_pattern
            
            return seasonal_patterns
            
        except Exception as e:
            logger.error(f"Error analyzing seasonal patterns: {e}")
            return {}
    
    def _analyze_daily_pattern(self, values: np.ndarray) -> Dict[str, Any]:
        """Analyze daily (24-hour) patterns."""
        try:
            if len(values) < 24:
                return {}
            
            # Reshape data into daily cycles
            daily_cycles = values[:len(values)//24*24].reshape(-1, 24)
            
            # Calculate hourly averages
            hourly_means = np.mean(daily_cycles, axis=0)
            hourly_stds = np.std(daily_cycles, axis=0)
            
            # Find peak and valley hours
            peak_hour = np.argmax(hourly_means)
            valley_hour = np.argmin(hourly_means)
            
            return {
                "hourly_means": hourly_means.tolist(),
                "hourly_stds": hourly_stds.tolist(),
                "peak_hour": int(peak_hour),
                "valley_hour": int(valley_hour),
                "amplitude": float(np.max(hourly_means) - np.min(hourly_means)),
                "consistency": float(1.0 - np.mean(hourly_stds) / np.mean(hourly_means))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing daily pattern: {e}")
            return {}
    
    def _analyze_weekly_pattern(self, values: np.ndarray) -> Dict[str, Any]:
        """Analyze weekly (168-hour) patterns."""
        try:
            if len(values) < 168:
                return {}
            
            # Reshape data into weekly cycles
            weekly_cycles = values[:len(values)//168*168].reshape(-1, 168)
            
            # Calculate daily averages
            daily_means = np.mean(weekly_cycles.reshape(-1, 7, 24), axis=(0, 2))
            daily_stds = np.std(weekly_cycles.reshape(-1, 7, 24), axis=(0, 2))
            
            # Find peak and valley days
            peak_day = np.argmax(daily_means)
            valley_day = np.argmin(daily_means)
            
            return {
                "daily_means": daily_means.tolist(),
                "daily_stds": daily_stds.tolist(),
                "peak_day": int(peak_day),
                "valley_day": int(valley_day),
                "amplitude": float(np.max(daily_means) - np.min(daily_means)),
                "consistency": float(1.0 - np.mean(daily_stds) / np.mean(daily_means))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing weekly pattern: {e}")
            return {}
    
    def _generate_forecast(self, df: pd.DataFrame, metric_name: str) -> List[Dict[str, Any]]:
        """Generate forecast for the trend."""
        try:
            if df.empty or len(df) < 3:
                return []
            
            # Simple linear forecast
            x = np.arange(len(df))
            y = df['value'].values
            
            slope, intercept, _, _, _ = stats.linregress(x, y)
            
            # Generate forecast points
            forecast = []
            last_timestamp = df.iloc[-1]['timestamp']
            
            for i in range(1, self.forecast_horizon + 1):
                forecast_x = len(df) + i - 1
                forecast_value = slope * forecast_x + intercept
                
                # Add some uncertainty
                std_err = np.std(y - (slope * x + intercept))
                confidence_interval = 1.96 * std_err  # 95% confidence
                
                forecast_timestamp = last_timestamp + timedelta(hours=i)
                
                forecast.append({
                    "timestamp": forecast_timestamp,
                    "value": forecast_value,
                    "lower_bound": forecast_value - confidence_interval,
                    "upper_bound": forecast_value + confidence_interval,
                    "confidence": 0.95
                })
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            return []
    
    def _has_cyclical_pattern(self, df: pd.DataFrame) -> bool:
        """Check if data has cyclical patterns."""
        try:
            if len(df) < 12:
                return False
            
            values = df['value'].values
            
            # Use autocorrelation to detect cycles
            autocorr = np.correlate(values, values, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Normalize
            autocorr = autocorr / autocorr[0]
            
            # Look for significant peaks (excluding lag 0)
            peaks, _ = find_peaks(autocorr[1:], height=0.3)
            
            return len(peaks) > 0
            
        except Exception as e:
            logger.error(f"Error checking cyclical pattern: {e}")
            return False
    
    def _has_seasonal_pattern(self, df: pd.DataFrame) -> bool:
        """Check if data has seasonal patterns."""
        try:
            if len(df) < 24:
                return False
            
            values = df['value'].values
            
            # Check for daily seasonality
            if len(values) >= 24:
                daily_cycles = values[:len(values)//24*24].reshape(-1, 24)
                daily_variance = np.var(daily_cycles, axis=0)
                
                # If variance is low, it suggests a consistent daily pattern
                if np.mean(daily_variance) < np.var(values) * 0.5:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking seasonal pattern: {e}")
            return False
    
    def _determine_overall_trend(self, analyses: List[TrendAnalysis]) -> TrendType:
        """Determine overall trend from multiple analyses."""
        try:
            if not analyses:
                return TrendType.IRREGULAR
            
            # Count trend types
            trend_counts = {}
            for analysis in analyses:
                trend_type = analysis.trend_type
                trend_counts[trend_type] = trend_counts.get(trend_type, 0) + 1
            
            # Return the most common trend type
            return max(trend_counts.items(), key=lambda x: x[1])[0]
            
        except Exception as e:
            logger.error(f"Error determining overall trend: {e}")
            return TrendType.IRREGULAR
    
    def _generate_trend_summary(self, analyses: List[TrendAnalysis]) -> Dict[str, Any]:
        """Generate summary of trend analyses."""
        try:
            if not analyses:
                return {}
            
            summary = {
                "total_analyses": len(analyses),
                "trend_distribution": {},
                "average_r_squared": 0.0,
                "total_anomalies": 0,
                "forecast_available": False
            }
            
            # Count trend types
            for analysis in analyses:
                trend_type = analysis.trend_type.value
                summary["trend_distribution"][trend_type] = summary["trend_distribution"].get(trend_type, 0) + 1
                summary["total_anomalies"] += len(analysis.anomalies)
                if analysis.forecast:
                    summary["forecast_available"] = True
            
            # Calculate average R-squared
            r_squared_values = [analysis.r_squared for analysis in analyses]
            summary["average_r_squared"] = statistics.mean(r_squared_values) if r_squared_values else 0.0
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating trend summary: {e}")
            return {}
    
    def _generate_trend_recommendations(self, analyses: List[TrendAnalysis], overall_trend: TrendType) -> List[str]:
        """Generate recommendations based on trend analysis."""
        try:
            recommendations = []
            
            # Overall trend recommendations
            if overall_trend == TrendType.DECREASING:
                recommendations.append("Quality is declining - investigate root causes and implement corrective measures")
            elif overall_trend == TrendType.INCREASING:
                recommendations.append("Quality is improving - maintain current practices and monitor for sustainability")
            elif overall_trend == TrendType.CYCLICAL:
                recommendations.append("Quality shows cyclical patterns - consider time-based quality controls")
            elif overall_trend == TrendType.SEASONAL:
                recommendations.append("Quality shows seasonal patterns - adjust monitoring frequency accordingly")
            
            # Anomaly recommendations
            total_anomalies = sum(len(analysis.anomalies) for analysis in analyses)
            if total_anomalies > 0:
                recommendations.append(f"Detected {total_anomalies} anomalies - investigate and address quality issues")
            
            # Forecast recommendations
            forecast_analyses = [analysis for analysis in analyses if analysis.forecast]
            if forecast_analyses:
                recommendations.append("Forecast data available - use for proactive quality management")
            
            # R-squared recommendations
            low_r_squared_analyses = [analysis for analysis in analyses if analysis.r_squared < 0.3]
            if low_r_squared_analyses:
                recommendations.append("Some metrics show weak trends - consider additional data collection")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating trend recommendations: {e}")
            return []
    
    def _compare_trend_types(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        """Compare trend types across sources."""
        try:
            comparison = {
                "trend_alignment": {},
                "consistency_score": 0.0,
                "dominant_trend": None
            }
            
            if not sources:
                return comparison
            
            # Count trend types
            trend_counts = {}
            for source_data in sources.values():
                trend_type = source_data.get("trend_type", "unknown")
                trend_counts[trend_type] = trend_counts.get(trend_type, 0) + 1
            
            # Determine dominant trend
            if trend_counts:
                dominant_trend = max(trend_counts.items(), key=lambda x: x[1])[0]
                comparison["dominant_trend"] = dominant_trend
                
                # Calculate consistency score
                total_sources = len(sources)
                dominant_count = trend_counts[dominant_trend]
                comparison["consistency_score"] = dominant_count / total_sources
            
            comparison["trend_alignment"] = trend_counts
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing trend types: {e}")
            return {}
    
    def _generate_comparison_summary(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparison summary."""
        try:
            if not sources:
                return {}
            
            summary = {
                "total_sources": len(sources),
                "average_slope": 0.0,
                "average_r_squared": 0.0,
                "total_anomalies": 0,
                "strongest_trend": None,
                "weakest_trend": None
            }
            
            # Calculate averages
            slopes = [source_data.get("slope", 0.0) for source_data in sources.values()]
            r_squared_values = [source_data.get("r_squared", 0.0) for source_data in sources.values()]
            anomaly_counts = [source_data.get("anomalies_count", 0) for source_data in sources.values()]
            
            summary["average_slope"] = statistics.mean(slopes) if slopes else 0.0
            summary["average_r_squared"] = statistics.mean(r_squared_values) if r_squared_values else 0.0
            summary["total_anomalies"] = sum(anomaly_counts)
            
            # Find strongest and weakest trends
            if r_squared_values:
                max_r_squared = max(r_squared_values)
                min_r_squared = min(r_squared_values)
                
                for source_name, source_data in sources.items():
                    if source_data.get("r_squared", 0.0) == max_r_squared:
                        summary["strongest_trend"] = source_name
                    if source_data.get("r_squared", 0.0) == min_r_squared:
                        summary["weakest_trend"] = source_name
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating comparison summary: {e}")
            return {}
    
    def _create_insufficient_data_analysis(self, source_name: str, metric_name: str) -> TrendAnalysis:
        """Create analysis for insufficient data."""
        return TrendAnalysis(
            source_name=source_name,
            metric_name=metric_name,
            trend_type=TrendType.IRREGULAR,
            trend_strength=TrendStrength.WEAK,
            slope=0.0,
            r_squared=0.0,
            p_value=1.0,
            confidence_interval=(0.0, 0.0)
        )
    
    def _create_no_data_report(self, source_name: str, period_start: datetime, period_end: datetime) -> QualityTrendReport:
        """Create report for no data available."""
        return QualityTrendReport(
            report_id=f"{source_name}_trend_{int(datetime.now().timestamp())}",
            source_name=source_name,
            period_start=period_start,
            period_end=period_end,
            overall_trend=TrendType.IRREGULAR,
            summary={"message": "No trend data available for the specified period"},
            recommendations=["Collect more quality data to enable trend analysis"]
        )
