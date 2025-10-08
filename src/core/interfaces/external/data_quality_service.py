"""
Data quality service interface for PBF-LB/M operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime

from ...domain.value_objects import QualityMetrics, DefectClassification
from ...domain.enums import QualityTier, DefectType


class DataQualityService(ABC):
    """
    Interface for data quality services in PBF-LB/M operations.
    
    This interface defines the contract for data quality assessment,
    validation, and improvement services.
    """
    
    @abstractmethod
    async def assess_data_quality(self, data: Dict[str, Any], quality_rules: Optional[Dict[str, Any]] = None) -> QualityMetrics:
        """
        Assess the quality of data.
        
        Args:
            data: The data to assess
            quality_rules: Optional quality rules to apply
            
        Returns:
            Quality metrics for the data
            
        Raises:
            DataQualityException: If assessment fails
        """
        pass
    
    @abstractmethod
    async def validate_data(self, data: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate data against schema or rules.
        
        Args:
            data: The data to validate
            schema: Optional schema to validate against
            
        Returns:
            Validation results
            
        Raises:
            ValidationException: If validation fails
        """
        pass
    
    @abstractmethod
    async def detect_anomalies(self, data: Dict[str, Any], anomaly_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Detect anomalies in data.
        
        Args:
            data: The data to analyze
            anomaly_config: Optional anomaly detection configuration
            
        Returns:
            List of detected anomalies
            
        Raises:
            AnomalyDetectionException: If detection fails
        """
        pass
    
    @abstractmethod
    async def classify_defects(self, data: Dict[str, Any], defect_data: Dict[str, Any]) -> List[DefectClassification]:
        """
        Classify defects in data.
        
        Args:
            data: The data containing defect information
            defect_data: Additional defect data for classification
            
        Returns:
            List of classified defects
            
        Raises:
            DefectClassificationException: If classification fails
        """
        pass
    
    @abstractmethod
    async def calculate_quality_score(self, quality_metrics: QualityMetrics) -> float:
        """
        Calculate overall quality score from metrics.
        
        Args:
            quality_metrics: The quality metrics to score
            
        Returns:
            Overall quality score (0-100)
            
        Raises:
            QualityScoringException: If scoring fails
        """
        pass
    
    @abstractmethod
    async def determine_quality_tier(self, quality_score: float) -> QualityTier:
        """
        Determine quality tier from score.
        
        Args:
            quality_score: The quality score to tier
            
        Returns:
            Quality tier
            
        Raises:
            QualityTieringException: If tiering fails
        """
        pass
    
    @abstractmethod
    async def get_quality_recommendations(self, quality_metrics: QualityMetrics) -> List[str]:
        """
        Get recommendations for improving data quality.
        
        Args:
            quality_metrics: The quality metrics to analyze
            
        Returns:
            List of improvement recommendations
            
        Raises:
            RecommendationException: If recommendation generation fails
        """
        pass
    
    @abstractmethod
    async def get_quality_trends(self, data_points: List[QualityMetrics], time_period: str = "month") -> Dict[str, Any]:
        """
        Get quality trends over time.
        
        Args:
            data_points: List of quality metrics over time
            time_period: Time period for trend analysis
            
        Returns:
            Dictionary containing quality trends
            
        Raises:
            TrendAnalysisException: If trend analysis fails
        """
        pass
    
    @abstractmethod
    async def get_quality_benchmarks(self, data_type: str) -> Dict[str, Any]:
        """
        Get quality benchmarks for a specific data type.
        
        Args:
            data_type: The type of data to get benchmarks for
            
        Returns:
            Dictionary containing quality benchmarks
            
        Raises:
            BenchmarkException: If benchmark retrieval fails
        """
        pass
    
    @abstractmethod
    async def compare_quality(self, metrics1: QualityMetrics, metrics2: QualityMetrics) -> Dict[str, Any]:
        """
        Compare two quality metrics.
        
        Args:
            metrics1: First quality metrics
            metrics2: Second quality metrics
            
        Returns:
            Dictionary containing comparison results
            
        Raises:
            QualityComparisonException: If comparison fails
        """
        pass
    
    @abstractmethod
    async def get_quality_thresholds(self, data_type: str) -> Dict[str, Any]:
        """
        Get quality thresholds for a specific data type.
        
        Args:
            data_type: The type of data to get thresholds for
            
        Returns:
            Dictionary containing quality thresholds
            
        Raises:
            ThresholdException: If threshold retrieval fails
        """
        pass
    
    @abstractmethod
    async def set_quality_thresholds(self, data_type: str, thresholds: Dict[str, Any]) -> bool:
        """
        Set quality thresholds for a specific data type.
        
        Args:
            data_type: The type of data to set thresholds for
            thresholds: The thresholds to set
            
        Returns:
            True if thresholds set successfully
            
        Raises:
            ThresholdException: If threshold setting fails
        """
        pass
    
    @abstractmethod
    async def get_quality_rules(self, data_type: str) -> List[Dict[str, Any]]:
        """
        Get quality rules for a specific data type.
        
        Args:
            data_type: The type of data to get rules for
            
        Returns:
            List of quality rules
            
        Raises:
            RuleException: If rule retrieval fails
        """
        pass
    
    @abstractmethod
    async def set_quality_rules(self, data_type: str, rules: List[Dict[str, Any]]) -> bool:
        """
        Set quality rules for a specific data type.
        
        Args:
            data_type: The type of data to set rules for
            rules: The rules to set
            
        Returns:
            True if rules set successfully
            
        Raises:
            RuleException: If rule setting fails
        """
        pass
    
    @abstractmethod
    async def get_quality_statistics(self, data_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get quality statistics for data.
        
        Args:
            data_type: Optional data type to filter statistics
            
        Returns:
            Dictionary containing quality statistics
            
        Raises:
            StatisticsException: If statistics retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_quality_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get quality alerts.
        
        Args:
            severity: Optional severity level to filter alerts
            
        Returns:
            List of quality alerts
            
        Raises:
            AlertException: If alert retrieval fails
        """
        pass
    
    @abstractmethod
    async def create_quality_alert(self, alert_data: Dict[str, Any]) -> str:
        """
        Create a quality alert.
        
        Args:
            alert_data: The alert data to create
            
        Returns:
            Alert ID
            
        Raises:
            AlertException: If alert creation fails
        """
        pass
    
    @abstractmethod
    async def resolve_quality_alert(self, alert_id: str, resolution_data: Dict[str, Any]) -> bool:
        """
        Resolve a quality alert.
        
        Args:
            alert_id: The ID of the alert to resolve
            resolution_data: The resolution data
            
        Returns:
            True if alert resolved successfully
            
        Raises:
            AlertException: If alert resolution fails
        """
        pass
    
    @abstractmethod
    async def get_quality_dashboard_data(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get data for quality dashboard.
        
        Args:
            filters: Optional filters to apply
            
        Returns:
            Dictionary containing dashboard data
            
        Raises:
            DashboardException: If dashboard data retrieval fails
        """
        pass
    
    @abstractmethod
    async def export_quality_report(self, report_config: Dict[str, Any]) -> str:
        """
        Export quality report.
        
        Args:
            report_config: Configuration for the report
            
        Returns:
            Report file path or URL
            
        Raises:
            ReportException: If report export fails
        """
        pass
    
    @abstractmethod
    async def get_service_health(self) -> Dict[str, Any]:
        """
        Get health status of the data quality service.
        
        Returns:
            Dictionary containing health status
        """
        pass
    
    @abstractmethod
    async def get_service_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics of the data quality service.
        
        Returns:
            Dictionary containing service metrics
        """
        pass
