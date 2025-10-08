"""
Anomaly Detector for PBF-LB/M Data Pipeline

This module provides anomaly detection capabilities for identifying
unusual patterns and outliers in PBF-LB/M manufacturing data.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy import stats

from .data_quality_service import QualityResult, QualityRule

logger = logging.getLogger(__name__)


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection."""
    contamination: float = 0.1  # Expected proportion of outliers
    random_state: int = 42
    n_estimators: int = 100
    max_samples: Union[str, int] = 'auto'
    max_features: Union[str, int] = 1.0
    bootstrap: bool = False
    n_jobs: int = -1
    verbose: int = 0
    warm_start: bool = False
    enable_statistical_detection: bool = True
    enable_clustering_detection: bool = True
    enable_isolation_forest: bool = True
    z_score_threshold: float = 3.0
    iqr_multiplier: float = 1.5


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    total_records: int
    anomaly_count: int
    anomaly_percentage: float
    anomaly_indices: List[int]
    anomaly_scores: List[float]
    anomaly_types: List[str]
    quality_results: List[QualityResult]
    detection_method: str
    processing_time_seconds: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AnomalyDetector:
    """
    Anomaly detector for PBF-LB/M manufacturing data.
    
    This detector uses multiple algorithms to identify anomalies in
    PBF-LB/M data including statistical methods, clustering, and
    isolation forest techniques.
    """
    
    def __init__(self, config: Optional[AnomalyConfig] = None):
        """
        Initialize the anomaly detector.
        
        Args:
            config: Anomaly detection configuration
        """
        self.config = config or AnomalyConfig()
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.dbscan = None
        
        logger.info("Anomaly Detector initialized")
    
    def detect_anomalies(self, data: List[Dict[str, Any]], 
                        data_type: str = 'generic') -> AnomalyResult:
        """
        Detect anomalies in the provided data.
        
        Args:
            data: List of data records
            data_type: Type of data (ispm, powder_bed, pbf_process, ct_scan)
            
        Returns:
            AnomalyResult: Anomaly detection result
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting anomaly detection for {len(data)} records of type {data_type}")
            
            # Convert data to DataFrame for easier processing
            df = pd.DataFrame(data)
            
            if df.empty:
                return AnomalyResult(
                    total_records=0,
                    anomaly_count=0,
                    anomaly_percentage=0.0,
                    anomaly_indices=[],
                    anomaly_scores=[],
                    anomaly_types=[],
                    quality_results=[],
                    detection_method='none',
                    processing_time_seconds=0.0
                )
            
            # Extract numerical features for anomaly detection
            numerical_features = self._extract_numerical_features(df, data_type)
            
            if numerical_features.empty:
                logger.warning("No numerical features found for anomaly detection")
                return AnomalyResult(
                total_records=len(data),
                    anomaly_count=0,
                    anomaly_percentage=0.0,
                    anomaly_indices=[],
                    anomaly_scores=[],
                    anomaly_types=[],
                    quality_results=[],
                    detection_method='none',
                    processing_time_seconds=(datetime.now() - start_time).total_seconds()
                )
            
            # Perform anomaly detection using multiple methods
            anomaly_results = []
            
            # Statistical anomaly detection
            if self.config.enable_statistical_detection:
                stat_anomalies = self._detect_statistical_anomalies(numerical_features)
                anomaly_results.append(('statistical', stat_anomalies))
            
            # Clustering-based anomaly detection
            if self.config.enable_clustering_detection:
                cluster_anomalies = self._detect_clustering_anomalies(numerical_features)
                anomaly_results.append(('clustering', cluster_anomalies))
            
            # Isolation Forest anomaly detection
            if self.config.enable_isolation_forest:
                isolation_anomalies = self._detect_isolation_forest_anomalies(numerical_features)
                anomaly_results.append(('isolation_forest', isolation_anomalies))
            
            # Combine results from different methods
            combined_anomalies = self._combine_anomaly_results(anomaly_results)
            
            # Create quality results
            quality_results = self._create_quality_results(data, combined_anomalies, data_type)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = AnomalyResult(
                total_records=len(data),
                anomaly_count=len(combined_anomalies['indices']),
                anomaly_percentage=(len(combined_anomalies['indices']) / len(data)) * 100,
                anomaly_indices=combined_anomalies['indices'],
                anomaly_scores=combined_anomalies['scores'],
                anomaly_types=combined_anomalies['types'],
                quality_results=quality_results,
                detection_method='combined',
                processing_time_seconds=processing_time,
                metadata={
                    'data_type': data_type,
                    'numerical_features_count': len(numerical_features.columns),
                    'detection_methods_used': [method for method, _ in anomaly_results],
                    'config': self.config.__dict__
                }
            )
            
            logger.info(f"Anomaly detection completed: {result.anomaly_count} anomalies found "
                       f"({result.anomaly_percentage:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return AnomalyResult(
                total_records=len(data),
                anomaly_count=0,
                anomaly_percentage=0.0,
                anomaly_indices=[],
                anomaly_scores=[],
                anomaly_types=[],
                quality_results=[],
                detection_method='error',
                processing_time_seconds=(datetime.now() - start_time).total_seconds(),
                metadata={'error': str(e)}
            )
    
    def _extract_numerical_features(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Extract numerical features for anomaly detection."""
        try:
            # Select numerical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove ID columns and other non-feature columns
            exclude_cols = ['id', 'record_id', 'timestamp', 'created_at', 'updated_at']
            numerical_cols = [col for col in numerical_cols if col.lower() not in exclude_cols]
            
            if not numerical_cols:
                return pd.DataFrame()
            
            # Extract features based on data type
            if data_type == 'ispm':
                # ISPM-specific features
                ispm_features = [col for col in numerical_cols if any(keyword in col.lower() 
                                for keyword in ['temperature', 'pressure', 'power', 'speed', 'energy'])]
                if ispm_features:
                    return df[ispm_features]
            
            elif data_type == 'powder_bed':
                # Powder bed-specific features
                bed_features = [col for col in numerical_cols if any(keyword in col.lower() 
                              for keyword in ['thickness', 'density', 'roughness', 'porosity'])]
                if bed_features:
                    return df[bed_features]
            
            elif data_type == 'pbf_process':
                # PBF process-specific features
                process_features = [col for col in numerical_cols if any(keyword in col.lower() 
                                   for keyword in ['laser_power', 'scan_speed', 'layer_thickness', 'hatch_spacing'])]
                if process_features:
                    return df[process_features]
            
            elif data_type == 'ct_scan':
                # CT scan-specific features
                ct_features = [col for col in numerical_cols if any(keyword in col.lower() 
                             for keyword in ['density', 'porosity', 'defect_size', 'voxel_value'])]
                if ct_features:
                    return df[ct_features]
            
            # Fallback to all numerical columns
            return df[numerical_cols]
            
        except Exception as e:
            logger.error(f"Error extracting numerical features: {e}")
            return pd.DataFrame()
    
    def _detect_statistical_anomalies(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies using statistical methods."""
        try:
            anomalies = {
                'indices': [],
                'scores': [],
                'types': []
            }
            
            for col in features.columns:
                # Z-score method
                z_scores = np.abs(stats.zscore(features[col].dropna()))
                z_anomalies = np.where(z_scores > self.config.z_score_threshold)[0]
                
                # IQR method
                Q1 = features[col].quantile(0.25)
                Q3 = features[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.config.iqr_multiplier * IQR
                upper_bound = Q3 + self.config.iqr_multiplier * IQR
                iqr_anomalies = np.where((features[col] < lower_bound) | (features[col] > upper_bound))[0]
                
                # Combine results
                col_anomalies = list(set(z_anomalies.tolist() + iqr_anomalies.tolist()))
                
                for idx in col_anomalies:
                    if idx not in anomalies['indices']:
                        anomalies['indices'].append(idx)
                        anomalies['scores'].append(max(z_scores[idx] if idx < len(z_scores) else 0, 1.0))
                        anomalies['types'].append(f'statistical_{col}')
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in statistical anomaly detection: {e}")
            return {'indices': [], 'scores': [], 'types': []}
    
    def _detect_clustering_anomalies(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies using clustering methods."""
        try:
            if len(features) < 10:  # Need minimum samples for clustering
                return {'indices': [], 'scores': [], 'types': []}
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features.fillna(features.mean()))
            
            # DBSCAN clustering
            self.dbscan = DBSCAN(eps=0.5, min_samples=5)
            clusters = self.dbscan.fit_predict(scaled_features)
            
            # Points labeled as -1 are outliers
            outlier_indices = np.where(clusters == -1)[0].tolist()
            
            return {
                'indices': outlier_indices,
                'scores': [1.0] * len(outlier_indices),
                'types': ['clustering'] * len(outlier_indices)
            }
            
        except Exception as e:
            logger.error(f"Error in clustering anomaly detection: {e}")
            return {'indices': [], 'scores': [], 'types': []}
    
    def _detect_isolation_forest_anomalies(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies using Isolation Forest."""
        try:
            if len(features) < 10:  # Need minimum samples for isolation forest
                return {'indices': [], 'scores': [], 'types': []}
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features.fillna(features.mean()))
            
            # Isolation Forest
            self.isolation_forest = IsolationForest(
                contamination=self.config.contamination,
                random_state=self.config.random_state,
                n_estimators=self.config.n_estimators
            )
            
            anomaly_labels = self.isolation_forest.fit_predict(scaled_features)
            anomaly_scores = self.isolation_forest.decision_function(scaled_features)
            
            # Points labeled as -1 are outliers
            outlier_indices = np.where(anomaly_labels == -1)[0].tolist()
            outlier_scores = [-anomaly_scores[i] for i in outlier_indices]  # Negative scores for outliers
            
            return {
                'indices': outlier_indices,
                'scores': outlier_scores,
                'types': ['isolation_forest'] * len(outlier_indices)
            }
            
        except Exception as e:
            logger.error(f"Error in isolation forest anomaly detection: {e}")
            return {'indices': [], 'scores': [], 'types': []}
    
    def _combine_anomaly_results(self, anomaly_results: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """Combine results from different anomaly detection methods."""
        try:
            combined = {
                'indices': [],
                'scores': [],
                'types': []
            }
            
            # Collect all anomalies
            for method, result in anomaly_results:
                combined['indices'].extend(result['indices'])
                combined['scores'].extend(result['scores'])
                combined['types'].extend(result['types'])
            
            # Remove duplicates while preserving order
            seen_indices = set()
            unique_anomalies = {
                'indices': [],
                'scores': [],
                'types': []
            }
            
            for i, idx in enumerate(combined['indices']):
                if idx not in seen_indices:
                    seen_indices.add(idx)
                    unique_anomalies['indices'].append(idx)
                    unique_anomalies['scores'].append(combined['scores'][i])
                    unique_anomalies['types'].append(combined['types'][i])
            
            return unique_anomalies
            
        except Exception as e:
            logger.error(f"Error combining anomaly results: {e}")
            return {'indices': [], 'scores': [], 'types': []}
    
    def _create_quality_results(self, data: List[Dict[str, Any]], 
                              anomalies: Dict[str, Any], 
                              data_type: str) -> List[QualityResult]:
        """Create quality results from anomaly detection."""
        try:
            quality_results = []
            
            for i, (idx, score, anomaly_type) in enumerate(zip(anomalies['indices'], 
                                                               anomalies['scores'], 
                                                               anomalies['types'])):
                if idx < len(data):
                    quality_result = QualityResult(
                        rule_id=f"anomaly_detection_{i}",
                        rule_name=f"Anomaly Detection - {anomaly_type}",
                        rule_type="anomaly",
                        passed=False,
                        quality_score=1.0 - min(score, 1.0),  # Convert to quality score
                        message=f"Anomaly detected using {anomaly_type} method (score: {score:.3f})",
                        record_id=str(idx),
                        field_name="multiple",
                        expected_value="normal_range",
                        actual_value=f"anomaly_score_{score:.3f}",
                        severity="high",
                        metadata={
                            'anomaly_type': anomaly_type,
                            'anomaly_score': score,
                            'data_type': data_type,
                            'detection_timestamp': datetime.now().isoformat()
                        }
                    )
                    quality_results.append(quality_result)
            
            return quality_results
            
        except Exception as e:
            logger.error(f"Error creating quality results: {e}")
            return []
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the anomaly detector."""
        try:
            return {
                'status': 'healthy',
                'config': self.config.__dict__,
                'models_initialized': {
                    'isolation_forest': self.isolation_forest is not None,
                    'scaler': self.scaler is not None,
                    'dbscan': self.dbscan is not None
                },
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# Convenience functions
def create_anomaly_detector(**kwargs) -> AnomalyDetector:
    """Create an anomaly detector with custom configuration."""
    config = AnomalyConfig(**kwargs)
    return AnomalyDetector(config)


def detect_anomalies(data: List[Dict[str, Any]], data_type: str = 'generic', **kwargs) -> AnomalyResult:
    """Convenience function for anomaly detection."""
    detector = create_anomaly_detector(**kwargs)
    return detector.detect_anomalies(data, data_type)