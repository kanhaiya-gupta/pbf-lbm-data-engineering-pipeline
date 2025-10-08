"""
Quality Scorer

This module provides quality scoring capabilities for the PBF-LB/M data pipeline.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import statistics

from src.data_pipeline.config.pipeline_config import get_pipeline_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScoringMethod(Enum):
    """Scoring method enumeration."""
    WEIGHTED_AVERAGE = "weighted_average"
    GEOMETRIC_MEAN = "geometric_mean"
    HARMONIC_MEAN = "harmonic_mean"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    CUSTOM = "custom"

class QualityDimension(Enum):
    """Quality dimension enumeration."""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    RELEVANCE = "relevance"
    ACCESSIBILITY = "accessibility"

@dataclass
class QualityScore:
    """Quality score data class."""
    source_name: str
    dimension: QualityDimension
    score: float
    weight: float
    method: ScoringMethod
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class OverallQualityScore:
    """Overall quality score data class."""
    source_name: str
    overall_score: float
    dimension_scores: List[QualityScore] = field(default_factory=list)
    scoring_method: ScoringMethod = ScoringMethod.WEIGHTED_AVERAGE
    confidence_level: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class QualityScoreHistory:
    """Quality score history data class."""
    source_name: str
    scores: List[OverallQualityScore] = field(default_factory=list)
    trend: str = "stable"  # improving, declining, stable
    trend_score: float = 0.0
    volatility: float = 0.0

class QualityScorer:
    """
    Quality scoring service for PBF-LB/M data pipeline.
    """
    
    def __init__(self):
        self.config = get_pipeline_config()
        self.quality_weights: Dict[QualityDimension, float] = {
            QualityDimension.COMPLETENESS: 0.20,
            QualityDimension.ACCURACY: 0.25,
            QualityDimension.CONSISTENCY: 0.15,
            QualityDimension.TIMELINESS: 0.15,
            QualityDimension.VALIDITY: 0.15,
            QualityDimension.UNIQUENESS: 0.10
        }
        self.score_history: Dict[str, QualityScoreHistory] = {}
        self.scoring_methods: Dict[ScoringMethod, callable] = {
            ScoringMethod.WEIGHTED_AVERAGE: self._calculate_weighted_average,
            ScoringMethod.GEOMETRIC_MEAN: self._calculate_geometric_mean,
            ScoringMethod.HARMONIC_MEAN: self._calculate_harmonic_mean,
            ScoringMethod.MINIMUM: self._calculate_minimum,
            ScoringMethod.MAXIMUM: self._calculate_maximum
        }
        
    def calculate_quality_score(self, source_name: str, dimension_scores: Dict[QualityDimension, float],
                              method: ScoringMethod = ScoringMethod.WEIGHTED_AVERAGE) -> OverallQualityScore:
        """
        Calculate overall quality score for a data source.
        
        Args:
            source_name: The data source name
            dimension_scores: Dictionary of dimension scores
            method: The scoring method to use
            
        Returns:
            OverallQualityScore: The calculated quality score
        """
        try:
            logger.info(f"Calculating quality score for {source_name} using {method.value} method")
            
            # Create quality score objects
            quality_scores = []
            for dimension, score in dimension_scores.items():
                weight = self.quality_weights.get(dimension, 0.0)
                quality_score = QualityScore(
                    source_name=source_name,
                    dimension=dimension,
                    score=score,
                    weight=weight,
                    method=method
                )
                quality_scores.append(quality_score)
            
            # Calculate overall score
            scoring_function = self.scoring_methods.get(method, self._calculate_weighted_average)
            overall_score = scoring_function(quality_scores)
            
            # Calculate confidence level
            confidence_level = self._calculate_confidence_level(quality_scores)
            
            # Create overall quality score
            overall_quality_score = OverallQualityScore(
                source_name=source_name,
                overall_score=overall_score,
                dimension_scores=quality_scores,
                scoring_method=method,
                confidence_level=confidence_level
            )
            
            # Store in history
            self._store_score_history(source_name, overall_quality_score)
            
            logger.info(f"Quality score calculated for {source_name}: {overall_score:.3f}")
            return overall_quality_score
            
        except Exception as e:
            logger.error(f"Error calculating quality score for {source_name}: {e}")
            raise
    
    def calculate_pbf_process_quality_score(self, data: List[Dict[str, Any]]) -> OverallQualityScore:
        """
        Calculate quality score for PBF process data.
        
        Args:
            data: List of PBF process data records
            
        Returns:
            OverallQualityScore: The calculated quality score
        """
        try:
            logger.info(f"Calculating PBF process quality score for {len(data)} records")
            
            # Calculate dimension scores
            dimension_scores = self._calculate_pbf_process_dimension_scores(data)
            
            return self.calculate_quality_score("pbf_process", dimension_scores)
            
        except Exception as e:
            logger.error(f"Error calculating PBF process quality score: {e}")
            raise
    
    def calculate_ispm_monitoring_quality_score(self, data: List[Dict[str, Any]]) -> OverallQualityScore:
        """
        Calculate quality score for ISPM monitoring data.
        
        Args:
            data: List of ISPM monitoring data records
            
        Returns:
            OverallQualityScore: The calculated quality score
        """
        try:
            logger.info(f"Calculating ISPM monitoring quality score for {len(data)} records")
            
            # Calculate dimension scores
            dimension_scores = self._calculate_ispm_monitoring_dimension_scores(data)
            
            return self.calculate_quality_score("ispm_monitoring", dimension_scores)
            
        except Exception as e:
            logger.error(f"Error calculating ISPM monitoring quality score: {e}")
            raise
    
    def calculate_ct_scan_quality_score(self, data: List[Dict[str, Any]]) -> OverallQualityScore:
        """
        Calculate quality score for CT scan data.
        
        Args:
            data: List of CT scan data records
            
        Returns:
            OverallQualityScore: The calculated quality score
        """
        try:
            logger.info(f"Calculating CT scan quality score for {len(data)} records")
            
            # Calculate dimension scores
            dimension_scores = self._calculate_ct_scan_dimension_scores(data)
            
            return self.calculate_quality_score("ct_scan", dimension_scores)
            
        except Exception as e:
            logger.error(f"Error calculating CT scan quality score: {e}")
            raise
    
    def calculate_powder_bed_quality_score(self, data: List[Dict[str, Any]]) -> OverallQualityScore:
        """
        Calculate quality score for powder bed data.
        
        Args:
            data: List of powder bed data records
            
        Returns:
            OverallQualityScore: The calculated quality score
        """
        try:
            logger.info(f"Calculating powder bed quality score for {len(data)} records")
            
            # Calculate dimension scores
            dimension_scores = self._calculate_powder_bed_dimension_scores(data)
            
            return self.calculate_quality_score("powder_bed", dimension_scores)
            
        except Exception as e:
            logger.error(f"Error calculating powder bed quality score: {e}")
            raise
    
    def get_quality_trend(self, source_name: str, days: int = 7) -> Dict[str, Any]:
        """
        Get quality trend for a data source.
        
        Args:
            source_name: The data source name
            days: Number of days to analyze
            
        Returns:
            Dict[str, Any]: Quality trend information
        """
        try:
            if source_name not in self.score_history:
                return {"trend": "no_data", "message": "No quality score history available"}
            
            history = self.score_history[source_name]
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Filter recent scores
            recent_scores = [score for score in history.scores if score.timestamp >= cutoff_date]
            
            if len(recent_scores) < 2:
                return {"trend": "insufficient_data", "message": "Insufficient data for trend analysis"}
            
            # Calculate trend
            scores = [score.overall_score for score in recent_scores]
            trend = self._calculate_trend(scores)
            volatility = self._calculate_volatility(scores)
            
            return {
                "trend": trend,
                "trend_score": self._calculate_trend_score(scores),
                "volatility": volatility,
                "data_points": len(recent_scores),
                "latest_score": scores[-1],
                "average_score": statistics.mean(scores),
                "score_range": {"min": min(scores), "max": max(scores)},
                "period_days": days
            }
            
        except Exception as e:
            logger.error(f"Error getting quality trend for {source_name}: {e}")
            return {"trend": "error", "message": str(e)}
    
    def compare_quality_scores(self, source_names: List[str]) -> Dict[str, Any]:
        """
        Compare quality scores across multiple sources.
        
        Args:
            source_names: List of source names to compare
            
        Returns:
            Dict[str, Any]: Comparison results
        """
        try:
            logger.info(f"Comparing quality scores for {len(source_names)} sources")
            
            comparison = {
                "sources": {},
                "rankings": [],
                "summary": {}
            }
            
            # Get latest scores for each source
            latest_scores = {}
            for source_name in source_names:
                if source_name in self.score_history and self.score_history[source_name].scores:
                    latest_score = self.score_history[source_name].scores[-1]
                    latest_scores[source_name] = latest_score
                    comparison["sources"][source_name] = {
                        "overall_score": latest_score.overall_score,
                        "confidence_level": latest_score.confidence_level,
                        "scoring_method": latest_score.scoring_method.value,
                        "timestamp": latest_score.timestamp.isoformat()
                    }
            
            # Create rankings
            rankings = sorted(latest_scores.items(), key=lambda x: x[1].overall_score, reverse=True)
            comparison["rankings"] = [
                {
                    "rank": i + 1,
                    "source_name": source_name,
                    "score": score.overall_score
                }
                for i, (source_name, score) in enumerate(rankings)
            ]
            
            # Calculate summary statistics
            if latest_scores:
                scores = [score.overall_score for score in latest_scores.values()]
                comparison["summary"] = {
                    "total_sources": len(latest_scores),
                    "average_score": statistics.mean(scores),
                    "median_score": statistics.median(scores),
                    "score_range": {"min": min(scores), "max": max(scores)},
                    "standard_deviation": statistics.stdev(scores) if len(scores) > 1 else 0.0
                }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing quality scores: {e}")
            return {"error": str(e)}
    
    def set_quality_weights(self, weights: Dict[QualityDimension, float]) -> bool:
        """
        Set quality dimension weights.
        
        Args:
            weights: Dictionary of dimension weights
            
        Returns:
            bool: True if weights were set successfully, False otherwise
        """
        try:
            # Validate weights
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.01:  # Allow small floating point errors
                logger.warning(f"Quality weights sum to {total_weight}, not 1.0")
            
            self.quality_weights.update(weights)
            logger.info("Updated quality dimension weights")
            return True
            
        except Exception as e:
            logger.error(f"Error setting quality weights: {e}")
            return False
    
    def get_quality_weights(self) -> Dict[QualityDimension, float]:
        """
        Get current quality dimension weights.
        
        Returns:
            Dict[QualityDimension, float]: Current quality weights
        """
        return self.quality_weights.copy()
    
    def _calculate_pbf_process_dimension_scores(self, data: List[Dict[str, Any]]) -> Dict[QualityDimension, float]:
        """Calculate dimension scores for PBF process data."""
        try:
            df = pd.DataFrame(data)
            dimension_scores = {}
            
            # Completeness
            dimension_scores[QualityDimension.COMPLETENESS] = self._calculate_completeness_score(df)
            
            # Accuracy
            dimension_scores[QualityDimension.ACCURACY] = self._calculate_accuracy_score(df, "pbf_process")
            
            # Consistency
            dimension_scores[QualityDimension.CONSISTENCY] = self._calculate_consistency_score(df, "pbf_process")
            
            # Timeliness
            dimension_scores[QualityDimension.TIMELINESS] = self._calculate_timeliness_score(df)
            
            # Validity
            dimension_scores[QualityDimension.VALIDITY] = self._calculate_validity_score(df, "pbf_process")
            
            # Uniqueness
            dimension_scores[QualityDimension.UNIQUENESS] = self._calculate_uniqueness_score(df)
            
            return dimension_scores
            
        except Exception as e:
            logger.error(f"Error calculating PBF process dimension scores: {e}")
            return {}
    
    def _calculate_ispm_monitoring_dimension_scores(self, data: List[Dict[str, Any]]) -> Dict[QualityDimension, float]:
        """Calculate dimension scores for ISPM monitoring data."""
        try:
            df = pd.DataFrame(data)
            dimension_scores = {}
            
            # Completeness
            dimension_scores[QualityDimension.COMPLETENESS] = self._calculate_completeness_score(df)
            
            # Accuracy
            dimension_scores[QualityDimension.ACCURACY] = self._calculate_accuracy_score(df, "ispm_monitoring")
            
            # Consistency
            dimension_scores[QualityDimension.CONSISTENCY] = self._calculate_consistency_score(df, "ispm_monitoring")
            
            # Timeliness
            dimension_scores[QualityDimension.TIMELINESS] = self._calculate_timeliness_score(df)
            
            # Validity
            dimension_scores[QualityDimension.VALIDITY] = self._calculate_validity_score(df, "ispm_monitoring")
            
            # Uniqueness
            dimension_scores[QualityDimension.UNIQUENESS] = self._calculate_uniqueness_score(df)
            
            return dimension_scores
            
        except Exception as e:
            logger.error(f"Error calculating ISPM monitoring dimension scores: {e}")
            return {}
    
    def _calculate_ct_scan_dimension_scores(self, data: List[Dict[str, Any]]) -> Dict[QualityDimension, float]:
        """Calculate dimension scores for CT scan data."""
        try:
            df = pd.DataFrame(data)
            dimension_scores = {}
            
            # Completeness
            dimension_scores[QualityDimension.COMPLETENESS] = self._calculate_completeness_score(df)
            
            # Accuracy
            dimension_scores[QualityDimension.ACCURACY] = self._calculate_accuracy_score(df, "ct_scan")
            
            # Consistency
            dimension_scores[QualityDimension.CONSISTENCY] = self._calculate_consistency_score(df, "ct_scan")
            
            # Timeliness
            dimension_scores[QualityDimension.TIMELINESS] = self._calculate_timeliness_score(df)
            
            # Validity
            dimension_scores[QualityDimension.VALIDITY] = self._calculate_validity_score(df, "ct_scan")
            
            # Uniqueness
            dimension_scores[QualityDimension.UNIQUENESS] = self._calculate_uniqueness_score(df)
            
            return dimension_scores
            
        except Exception as e:
            logger.error(f"Error calculating CT scan dimension scores: {e}")
            return {}
    
    def _calculate_powder_bed_dimension_scores(self, data: List[Dict[str, Any]]) -> Dict[QualityDimension, float]:
        """Calculate dimension scores for powder bed data."""
        try:
            df = pd.DataFrame(data)
            dimension_scores = {}
            
            # Completeness
            dimension_scores[QualityDimension.COMPLETENESS] = self._calculate_completeness_score(df)
            
            # Accuracy
            dimension_scores[QualityDimension.ACCURACY] = self._calculate_accuracy_score(df, "powder_bed")
            
            # Consistency
            dimension_scores[QualityDimension.CONSISTENCY] = self._calculate_consistency_score(df, "powder_bed")
            
            # Timeliness
            dimension_scores[QualityDimension.TIMELINESS] = self._calculate_timeliness_score(df)
            
            # Validity
            dimension_scores[QualityDimension.VALIDITY] = self._calculate_validity_score(df, "powder_bed")
            
            # Uniqueness
            dimension_scores[QualityDimension.UNIQUENESS] = self._calculate_uniqueness_score(df)
            
            return dimension_scores
            
        except Exception as e:
            logger.error(f"Error calculating powder bed dimension scores: {e}")
            return {}
    
    def _calculate_completeness_score(self, df: pd.DataFrame) -> float:
        """Calculate completeness score."""
        try:
            if df.empty:
                return 0.0
            
            # Calculate percentage of non-null values
            total_cells = df.size
            non_null_cells = df.count().sum()
            
            return non_null_cells / total_cells if total_cells > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating completeness score: {e}")
            return 0.0
    
    def _calculate_accuracy_score(self, df: pd.DataFrame, source_type: str) -> float:
        """Calculate accuracy score based on source type."""
        try:
            if df.empty:
                return 0.0
            
            # Source-specific accuracy calculations
            if source_type == "pbf_process":
                return self._calculate_pbf_accuracy_score(df)
            elif source_type == "ispm_monitoring":
                return self._calculate_ispm_accuracy_score(df)
            elif source_type == "ct_scan":
                return self._calculate_ct_accuracy_score(df)
            elif source_type == "powder_bed":
                return self._calculate_powder_bed_accuracy_score(df)
            else:
                return 0.95  # Default accuracy score
                
        except Exception as e:
            logger.error(f"Error calculating accuracy score: {e}")
            return 0.0
    
    def _calculate_pbf_accuracy_score(self, df: pd.DataFrame) -> float:
        """Calculate accuracy score for PBF process data."""
        try:
            accuracy_factors = []
            
            # Temperature accuracy (within expected ranges)
            if "chamber_temperature" in df.columns:
                temp_col = pd.to_numeric(df["chamber_temperature"], errors='coerce')
                valid_temps = temp_col[(temp_col >= 20) & (temp_col <= 1000)]
                temp_accuracy = len(valid_temps) / len(temp_col) if len(temp_col) > 0 else 0.0
                accuracy_factors.append(temp_accuracy)
            
            # Pressure accuracy
            if "chamber_pressure" in df.columns:
                pressure_col = pd.to_numeric(df["chamber_pressure"], errors='coerce')
                valid_pressures = pressure_col[(pressure_col >= 0) & (pressure_col <= 10)]
                pressure_accuracy = len(valid_pressures) / len(pressure_col) if len(pressure_col) > 0 else 0.0
                accuracy_factors.append(pressure_accuracy)
            
            return statistics.mean(accuracy_factors) if accuracy_factors else 0.95
            
        except Exception as e:
            logger.error(f"Error calculating PBF accuracy score: {e}")
            return 0.95
    
    def _calculate_ispm_accuracy_score(self, df: pd.DataFrame) -> float:
        """Calculate accuracy score for ISPM monitoring data."""
        try:
            accuracy_factors = []
            
            # Melt pool temperature accuracy
            if "melt_pool_temperature" in df.columns:
                temp_col = pd.to_numeric(df["melt_pool_temperature"], errors='coerce')
                valid_temps = temp_col[(temp_col >= 1000) & (temp_col <= 3000)]
                temp_accuracy = len(valid_temps) / len(temp_col) if len(temp_col) > 0 else 0.0
                accuracy_factors.append(temp_accuracy)
            
            # Plume intensity accuracy
            if "plume_intensity" in df.columns:
                plume_col = pd.to_numeric(df["plume_intensity"], errors='coerce')
                valid_plumes = plume_col[(plume_col >= 0) & (plume_col <= 100)]
                plume_accuracy = len(valid_plumes) / len(plume_col) if len(plume_col) > 0 else 0.0
                accuracy_factors.append(plume_accuracy)
            
            return statistics.mean(accuracy_factors) if accuracy_factors else 0.95
            
        except Exception as e:
            logger.error(f"Error calculating ISPM accuracy score: {e}")
            return 0.95
    
    def _calculate_ct_accuracy_score(self, df: pd.DataFrame) -> float:
        """Calculate accuracy score for CT scan data."""
        try:
            accuracy_factors = []
            
            # Porosity accuracy
            if "porosity_percentage" in df.columns:
                porosity_col = pd.to_numeric(df["porosity_percentage"], errors='coerce')
                valid_porosity = porosity_col[(porosity_col >= 0) & (porosity_col <= 100)]
                porosity_accuracy = len(valid_porosity) / len(porosity_col) if len(porosity_col) > 0 else 0.0
                accuracy_factors.append(porosity_accuracy)
            
            # Defect count accuracy
            if "num_defects" in df.columns:
                defects_col = pd.to_numeric(df["num_defects"], errors='coerce')
                valid_defects = defects_col[defects_col >= 0]
                defects_accuracy = len(valid_defects) / len(defects_col) if len(defects_col) > 0 else 0.0
                accuracy_factors.append(defects_accuracy)
            
            return statistics.mean(accuracy_factors) if accuracy_factors else 0.95
            
        except Exception as e:
            logger.error(f"Error calculating CT accuracy score: {e}")
            return 0.95
    
    def _calculate_powder_bed_accuracy_score(self, df: pd.DataFrame) -> float:
        """Calculate accuracy score for powder bed data."""
        try:
            accuracy_factors = []
            
            # Layer number accuracy
            if "layer_number" in df.columns:
                layer_col = pd.to_numeric(df["layer_number"], errors='coerce')
                valid_layers = layer_col[(layer_col >= 1) & (layer_col <= 5000)]
                layer_accuracy = len(valid_layers) / len(layer_col) if len(layer_col) > 0 else 0.0
                accuracy_factors.append(layer_accuracy)
            
            # Porosity metric accuracy
            if "porosity_metric" in df.columns:
                porosity_col = pd.to_numeric(df["porosity_metric"], errors='coerce')
                valid_porosity = porosity_col[(porosity_col >= 0) & (porosity_col <= 1)]
                porosity_accuracy = len(valid_porosity) / len(porosity_col) if len(porosity_col) > 0 else 0.0
                accuracy_factors.append(porosity_accuracy)
            
            return statistics.mean(accuracy_factors) if accuracy_factors else 0.95
            
        except Exception as e:
            logger.error(f"Error calculating powder bed accuracy score: {e}")
            return 0.95
    
    def _calculate_consistency_score(self, df: pd.DataFrame, source_type: str) -> float:
        """Calculate consistency score."""
        try:
            if df.empty:
                return 0.0
            
            consistency_factors = []
            
            # Check for consistent data types
            for column in df.columns:
                if df[column].dtype == 'object':
                    # Check if all values can be converted to the same type
                    try:
                        pd.to_numeric(df[column], errors='raise')
                        consistency_factors.append(1.0)
                    except:
                        consistency_factors.append(0.8)
                else:
                    consistency_factors.append(1.0)
            
            return statistics.mean(consistency_factors) if consistency_factors else 0.95
            
        except Exception as e:
            logger.error(f"Error calculating consistency score: {e}")
            return 0.95
    
    def _calculate_timeliness_score(self, df: pd.DataFrame) -> float:
        """Calculate timeliness score."""
        try:
            if df.empty:
                return 0.0
            
            # Look for timestamp columns
            timestamp_columns = [col for col in df.columns if 'timestamp' in col.lower() or 'time' in col.lower()]
            
            if not timestamp_columns:
                return 0.95  # Default timeliness score
            
            # Calculate timeliness based on data freshness
            latest_timestamp = None
            for col in timestamp_columns:
                try:
                    timestamps = pd.to_datetime(df[col], errors='coerce')
                    if not timestamps.isna().all():
                        col_latest = timestamps.max()
                        if latest_timestamp is None or col_latest > latest_timestamp:
                            latest_timestamp = col_latest
                except:
                    continue
            
            if latest_timestamp is None:
                return 0.95
            
            # Calculate freshness (data within last hour = 1.0, older = lower score)
            time_diff = datetime.now() - latest_timestamp
            hours_old = time_diff.total_seconds() / 3600
            
            if hours_old <= 1:
                return 1.0
            elif hours_old <= 24:
                return max(0.5, 1.0 - (hours_old - 1) / 23)
            else:
                return 0.1
            
        except Exception as e:
            logger.error(f"Error calculating timeliness score: {e}")
            return 0.95
    
    def _calculate_validity_score(self, df: pd.DataFrame, source_type: str) -> float:
        """Calculate validity score."""
        try:
            if df.empty:
                return 0.0
            
            # Source-specific validity checks
            validity_factors = []
            
            if source_type == "pbf_process":
                # Check machine ID format
                if "machine_id" in df.columns:
                    valid_ids = df["machine_id"].str.match(r"^PBF-[A-Z]{2}-\d{3}$", na=False)
                    validity_factors.append(valid_ids.mean())
            
            elif source_type == "ispm_monitoring":
                # Check sensor ID format
                if "sensor_id" in df.columns:
                    valid_ids = df["sensor_id"].str.match(r"^ISPM-\d{3}$", na=False)
                    validity_factors.append(valid_ids.mean())
            
            elif source_type == "ct_scan":
                # Check scan ID format
                if "scan_id" in df.columns:
                    valid_ids = df["scan_id"].str.match(r"^CT-\d{6}$", na=False)
                    validity_factors.append(valid_ids.mean())
            
            elif source_type == "powder_bed":
                # Check image ID format
                if "image_id" in df.columns:
                    valid_ids = df["image_id"].str.match(r"^PB-\d{8}$", na=False)
                    validity_factors.append(valid_ids.mean())
            
            return statistics.mean(validity_factors) if validity_factors else 0.95
            
        except Exception as e:
            logger.error(f"Error calculating validity score: {e}")
            return 0.95
    
    def _calculate_uniqueness_score(self, df: pd.DataFrame) -> float:
        """Calculate uniqueness score."""
        try:
            if df.empty:
                return 0.0
            
            # Check for duplicate rows
            total_rows = len(df)
            unique_rows = len(df.drop_duplicates())
            
            return unique_rows / total_rows if total_rows > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating uniqueness score: {e}")
            return 0.95
    
    def _calculate_weighted_average(self, quality_scores: List[QualityScore]) -> float:
        """Calculate weighted average of quality scores."""
        try:
            if not quality_scores:
                return 0.0
            
            weighted_sum = sum(score.score * score.weight for score in quality_scores)
            total_weight = sum(score.weight for score in quality_scores)
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating weighted average: {e}")
            return 0.0
    
    def _calculate_geometric_mean(self, quality_scores: List[QualityScore]) -> float:
        """Calculate geometric mean of quality scores."""
        try:
            if not quality_scores:
                return 0.0
            
            scores = [score.score for score in quality_scores if score.score > 0]
            if not scores:
                return 0.0
            
            return statistics.geometric_mean(scores)
            
        except Exception as e:
            logger.error(f"Error calculating geometric mean: {e}")
            return 0.0
    
    def _calculate_harmonic_mean(self, quality_scores: List[QualityScore]) -> float:
        """Calculate harmonic mean of quality scores."""
        try:
            if not quality_scores:
                return 0.0
            
            scores = [score.score for score in quality_scores if score.score > 0]
            if not scores:
                return 0.0
            
            return statistics.harmonic_mean(scores)
            
        except Exception as e:
            logger.error(f"Error calculating harmonic mean: {e}")
            return 0.0
    
    def _calculate_minimum(self, quality_scores: List[QualityScore]) -> float:
        """Calculate minimum of quality scores."""
        try:
            if not quality_scores:
                return 0.0
            
            return min(score.score for score in quality_scores)
            
        except Exception as e:
            logger.error(f"Error calculating minimum: {e}")
            return 0.0
    
    def _calculate_maximum(self, quality_scores: List[QualityScore]) -> float:
        """Calculate maximum of quality scores."""
        try:
            if not quality_scores:
                return 0.0
            
            return max(score.score for score in quality_scores)
            
        except Exception as e:
            logger.error(f"Error calculating maximum: {e}")
            return 0.0
    
    def _calculate_confidence_level(self, quality_scores: List[QualityScore]) -> float:
        """Calculate confidence level based on quality scores."""
        try:
            if not quality_scores:
                return 0.0
            
            # Confidence based on number of dimensions and score consistency
            num_dimensions = len(quality_scores)
            scores = [score.score for score in quality_scores]
            
            # Base confidence from number of dimensions
            base_confidence = min(1.0, num_dimensions / 6.0)
            
            # Adjust for score consistency (lower variance = higher confidence)
            if len(scores) > 1:
                variance = statistics.variance(scores)
                consistency_factor = max(0.5, 1.0 - variance)
            else:
                consistency_factor = 1.0
            
            return base_confidence * consistency_factor
            
        except Exception as e:
            logger.error(f"Error calculating confidence level: {e}")
            return 0.0
    
    def _store_score_history(self, source_name: str, overall_score: OverallQualityScore):
        """Store quality score in history."""
        try:
            if source_name not in self.score_history:
                self.score_history[source_name] = QualityScoreHistory(source_name=source_name)
            
            self.score_history[source_name].scores.append(overall_score)
            
            # Keep only last 100 scores
            if len(self.score_history[source_name].scores) > 100:
                self.score_history[source_name].scores = self.score_history[source_name].scores[-100:]
            
            # Update trend
            self._update_trend(source_name)
            
        except Exception as e:
            logger.error(f"Error storing score history: {e}")
    
    def _update_trend(self, source_name: str):
        """Update quality trend for a source."""
        try:
            if source_name not in self.score_history:
                return
            
            history = self.score_history[source_name]
            if len(history.scores) < 3:
                return
            
            # Get recent scores
            recent_scores = [score.overall_score for score in history.scores[-10:]]
            
            # Calculate trend
            trend = self._calculate_trend(recent_scores)
            trend_score = self._calculate_trend_score(recent_scores)
            volatility = self._calculate_volatility(recent_scores)
            
            history.trend = trend
            history.trend_score = trend_score
            history.volatility = volatility
            
        except Exception as e:
            logger.error(f"Error updating trend: {e}")
    
    def _calculate_trend(self, scores: List[float]) -> str:
        """Calculate trend from scores."""
        try:
            if len(scores) < 2:
                return "stable"
            
            # Simple linear regression
            x = list(range(len(scores)))
            y = scores
            
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            
            if slope > 0.01:
                return "improving"
            elif slope < -0.01:
                return "declining"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Error calculating trend: {e}")
            return "stable"
    
    def _calculate_trend_score(self, scores: List[float]) -> float:
        """Calculate trend score (-1 to 1)."""
        try:
            if len(scores) < 2:
                return 0.0
            
            # Calculate correlation with time
            x = list(range(len(scores)))
            y = scores
            
            correlation = np.corrcoef(x, y)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating trend score: {e}")
            return 0.0
    
    def _calculate_volatility(self, scores: List[float]) -> float:
        """Calculate volatility (standard deviation)."""
        try:
            if len(scores) < 2:
                return 0.0
            
            return statistics.stdev(scores)
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.0
