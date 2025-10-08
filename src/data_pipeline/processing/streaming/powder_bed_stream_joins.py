"""
Powder Bed Stream Joins

This module provides stream-to-stream join capabilities for powder bed monitoring data.
"""

import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta

from src.data_pipeline.config.pipeline_config import get_pipeline_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PowderBedStreamJoins:
    """
    Stream-to-stream joins for powder bed monitoring data.
    """
    
    def __init__(self):
        self.config = get_pipeline_config()
        self.join_windows = self._load_join_windows()
        self.stream_buffers = {}
    
    def _load_join_windows(self) -> Dict[str, int]:
        """Load join window configurations."""
        try:
            return self.config.get('powder_bed_join_windows', {
                'image_quality': 15,      # 15 seconds
                'layer_analysis': 30,     # 30 seconds
                'defect_detection': 45,   # 45 seconds
                'quality_assessment': 60  # 60 seconds
            })
        except Exception as e:
            logger.error(f"Error loading join windows: {e}")
            return {}
    
    def join_image_quality_streams(self, image_data: Dict[str, Any], 
                                 quality_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Join image and quality streams."""
        try:
            # Check if data is within join window
            if not self._is_within_join_window(image_data, quality_data, 'image_quality'):
                return None
            
            # Create joined record
            joined_data = {
                'timestamp': image_data.get('timestamp'),
                'process_id': image_data.get('process_id'),
                'layer_number': image_data.get('layer_number'),
                'image_metadata': {
                    'image_id': image_data.get('image_id'),
                    'resolution': image_data.get('resolution'),
                    'file_size': image_data.get('file_size'),
                    'format': image_data.get('format')
                },
                'quality_metrics': {
                    'sharpness': quality_data.get('sharpness'),
                    'contrast': quality_data.get('contrast'),
                    'brightness': quality_data.get('brightness'),
                    'noise_level': quality_data.get('noise_level')
                },
                'joined_at': datetime.now().isoformat()
            }
            
            # Calculate overall image quality score
            joined_data['overall_image_quality'] = self._calculate_image_quality_score(
                joined_data['quality_metrics']
            )
            
            # Determine image quality status
            joined_data['image_quality_status'] = self._determine_image_quality_status(
                joined_data['overall_image_quality']
            )
            
            logger.info("Image and quality streams joined successfully")
            return joined_data
            
        except Exception as e:
            logger.error(f"Error joining image and quality streams: {e}")
            return None
    
    def join_layer_analysis_streams(self, layer_data: Dict[str, Any], 
                                  analysis_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Join layer and analysis streams."""
        try:
            # Check if data is within join window
            if not self._is_within_join_window(layer_data, analysis_data, 'layer_analysis'):
                return None
            
            # Create joined record
            joined_data = {
                'timestamp': layer_data.get('timestamp'),
                'process_id': layer_data.get('process_id'),
                'layer_number': layer_data.get('layer_number'),
                'layer_parameters': {
                    'thickness': layer_data.get('thickness'),
                    'density': layer_data.get('density'),
                    'uniformity': layer_data.get('uniformity'),
                    'coverage': layer_data.get('coverage')
                },
                'analysis_results': {
                    'defect_count': analysis_data.get('defect_count'),
                    'defect_types': analysis_data.get('defect_types'),
                    'porosity': analysis_data.get('porosity'),
                    'surface_roughness': analysis_data.get('surface_roughness')
                },
                'joined_at': datetime.now().isoformat()
            }
            
            # Calculate layer quality score
            joined_data['layer_quality_score'] = self._calculate_layer_quality_score(
                joined_data['layer_parameters'], joined_data['analysis_results']
            )
            
            # Determine layer status
            joined_data['layer_status'] = self._determine_layer_status(
                joined_data['layer_quality_score']
            )
            
            logger.info("Layer and analysis streams joined successfully")
            return joined_data
            
        except Exception as e:
            logger.error(f"Error joining layer and analysis streams: {e}")
            return None
    
    def join_defect_detection_streams(self, defect_data: Dict[str, Any], 
                                    detection_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Join defect and detection streams."""
        try:
            # Check if data is within join window
            if not self._is_within_join_window(defect_data, detection_data, 'defect_detection'):
                return None
            
            # Create joined record
            joined_data = {
                'timestamp': defect_data.get('timestamp'),
                'process_id': defect_data.get('process_id'),
                'layer_number': defect_data.get('layer_number'),
                'defect_info': {
                    'defect_id': defect_data.get('defect_id'),
                    'defect_type': defect_data.get('defect_type'),
                    'severity': defect_data.get('severity'),
                    'location': defect_data.get('location'),
                    'size': defect_data.get('size')
                },
                'detection_metadata': {
                    'detection_method': detection_data.get('detection_method'),
                    'confidence_score': detection_data.get('confidence_score'),
                    'detection_time': detection_data.get('detection_time'),
                    'algorithm_version': detection_data.get('algorithm_version')
                },
                'joined_at': datetime.now().isoformat()
            }
            
            # Calculate defect risk score
            joined_data['defect_risk_score'] = self._calculate_defect_risk_score(
                joined_data['defect_info'], joined_data['detection_metadata']
            )
            
            # Determine defect action required
            joined_data['defect_action'] = self._determine_defect_action(
                joined_data['defect_risk_score']
            )
            
            logger.info("Defect and detection streams joined successfully")
            return joined_data
            
        except Exception as e:
            logger.error(f"Error joining defect and detection streams: {e}")
            return None
    
    def join_quality_assessment_streams(self, quality_data_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Join multiple quality assessment streams."""
        try:
            if len(quality_data_list) < 2:
                return None
            
            # Check if all data is within join window
            if not self._are_within_join_window(quality_data_list, 'quality_assessment'):
                return None
            
            # Group data by quality type
            quality_groups = {}
            for data in quality_data_list:
                quality_type = data.get('quality_type', 'unknown')
                if quality_type not in quality_groups:
                    quality_groups[quality_type] = []
                quality_groups[quality_type].append(data)
            
            # Create joined record
            joined_data = {
                'timestamp': quality_data_list[0].get('timestamp'),
                'process_id': quality_data_list[0].get('process_id'),
                'layer_number': quality_data_list[0].get('layer_number'),
                'joined_at': datetime.now().isoformat()
            }
            
            # Add quality data
            for quality_type, data_list in quality_groups.items():
                if data_list:
                    latest_data = max(data_list, key=lambda x: x.get('timestamp', ''))
                    joined_data[f'{quality_type}_score'] = latest_data.get('score')
                    joined_data[f'{quality_type}_status'] = latest_data.get('status')
            
            # Calculate overall quality score
            joined_data['overall_quality_score'] = self._calculate_overall_quality_score(
                quality_groups
            )
            
            # Determine overall quality status
            joined_data['overall_quality_status'] = self._determine_overall_quality_status(
                joined_data['overall_quality_score']
            )
            
            logger.info("Quality assessment streams joined successfully")
            return joined_data
            
        except Exception as e:
            logger.error(f"Error joining quality assessment streams: {e}")
            return None
    
    def _is_within_join_window(self, data1: Dict[str, Any], data2: Dict[str, Any], 
                              join_type: str) -> bool:
        """Check if two data records are within join window."""
        try:
            window_seconds = self.join_windows.get(join_type, 30)
            
            timestamp1 = self._parse_timestamp(data1.get('timestamp'))
            timestamp2 = self._parse_timestamp(data2.get('timestamp'))
            
            if not timestamp1 or not timestamp2:
                return False
            
            time_diff = abs((timestamp1 - timestamp2).total_seconds())
            return time_diff <= window_seconds
            
        except Exception as e:
            logger.error(f"Error checking join window: {e}")
            return False
    
    def _are_within_join_window(self, data_list: List[Dict[str, Any]], join_type: str) -> bool:
        """Check if multiple data records are within join window."""
        try:
            window_seconds = self.join_windows.get(join_type, 60)
            
            timestamps = []
            for data in data_list:
                timestamp = self._parse_timestamp(data.get('timestamp'))
                if timestamp:
                    timestamps.append(timestamp)
            
            if len(timestamps) < 2:
                return False
            
            # Check if all timestamps are within the window
            min_timestamp = min(timestamps)
            max_timestamp = max(timestamps)
            
            time_diff = (max_timestamp - min_timestamp).total_seconds()
            return time_diff <= window_seconds
            
        except Exception as e:
            logger.error(f"Error checking join window for multiple records: {e}")
            return False
    
    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse timestamp string to datetime object."""
        try:
            if isinstance(timestamp_str, str):
                return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return None
        except Exception as e:
            logger.error(f"Error parsing timestamp: {e}")
            return None
    
    def _calculate_image_quality_score(self, quality_metrics: Dict[str, Any]) -> float:
        """Calculate overall image quality score."""
        try:
            scores = []
            
            # Sharpness score (0-1)
            if 'sharpness' in quality_metrics:
                sharpness = quality_metrics['sharpness']
                sharpness_score = min(1.0, sharpness / 100.0)  # Assuming 100 is max
                scores.append(sharpness_score)
            
            # Contrast score (0-1)
            if 'contrast' in quality_metrics:
                contrast = quality_metrics['contrast']
                contrast_score = min(1.0, contrast / 100.0)  # Assuming 100 is max
                scores.append(contrast_score)
            
            # Brightness score (0-1)
            if 'brightness' in quality_metrics:
                brightness = quality_metrics['brightness']
                brightness_score = 1.0 - abs(brightness - 50.0) / 50.0  # 50 is optimal
                scores.append(brightness_score)
            
            # Noise level score (0-1, inverted)
            if 'noise_level' in quality_metrics:
                noise = quality_metrics['noise_level']
                noise_score = 1.0 - min(1.0, noise / 100.0)  # Lower noise is better
                scores.append(noise_score)
            
            return sum(scores) / len(scores) if scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating image quality score: {e}")
            return 0.0
    
    def _determine_image_quality_status(self, quality_score: float) -> str:
        """Determine image quality status based on score."""
        try:
            if quality_score >= 0.8:
                return 'excellent'
            elif quality_score >= 0.6:
                return 'good'
            elif quality_score >= 0.4:
                return 'fair'
            else:
                return 'poor'
        except Exception as e:
            logger.error(f"Error determining image quality status: {e}")
            return 'unknown'
    
    def _calculate_layer_quality_score(self, layer_params: Dict[str, Any], 
                                     analysis_results: Dict[str, Any]) -> float:
        """Calculate layer quality score."""
        try:
            scores = []
            
            # Layer parameter scores
            if 'thickness' in layer_params:
                thickness = layer_params['thickness']
                thickness_score = 1.0 - abs(thickness - 0.1) / 0.1  # 0.1mm is optimal
                scores.append(max(0.0, thickness_score))
            
            if 'density' in layer_params:
                density = layer_params['density']
                density_score = min(1.0, density / 100.0)  # 100% is optimal
                scores.append(density_score)
            
            if 'uniformity' in layer_params:
                uniformity = layer_params['uniformity']
                uniformity_score = min(1.0, uniformity / 100.0)  # 100% is optimal
                scores.append(uniformity_score)
            
            if 'coverage' in layer_params:
                coverage = layer_params['coverage']
                coverage_score = min(1.0, coverage / 100.0)  # 100% is optimal
                scores.append(coverage_score)
            
            # Analysis result scores
            if 'defect_count' in analysis_results:
                defect_count = analysis_results['defect_count']
                defect_score = 1.0 - min(1.0, defect_count / 10.0)  # 0 defects is optimal
                scores.append(defect_score)
            
            if 'porosity' in analysis_results:
                porosity = analysis_results['porosity']
                porosity_score = 1.0 - min(1.0, porosity / 10.0)  # 0% porosity is optimal
                scores.append(porosity_score)
            
            return sum(scores) / len(scores) if scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating layer quality score: {e}")
            return 0.0
    
    def _determine_layer_status(self, quality_score: float) -> str:
        """Determine layer status based on quality score."""
        try:
            if quality_score >= 0.8:
                return 'acceptable'
            elif quality_score >= 0.6:
                return 'marginal'
            else:
                return 'reject'
        except Exception as e:
            logger.error(f"Error determining layer status: {e}")
            return 'unknown'
    
    def _calculate_defect_risk_score(self, defect_info: Dict[str, Any], 
                                   detection_metadata: Dict[str, Any]) -> float:
        """Calculate defect risk score."""
        try:
            risk_factors = []
            
            # Severity factor
            if 'severity' in defect_info:
                severity = defect_info['severity']
                if severity == 'critical':
                    risk_factors.append(1.0)
                elif severity == 'high':
                    risk_factors.append(0.8)
                elif severity == 'medium':
                    risk_factors.append(0.6)
                elif severity == 'low':
                    risk_factors.append(0.4)
                else:
                    risk_factors.append(0.2)
            
            # Size factor
            if 'size' in defect_info:
                size = defect_info['size']
                size_factor = min(1.0, size / 100.0)  # Larger defects are riskier
                risk_factors.append(size_factor)
            
            # Confidence factor (inverted)
            if 'confidence_score' in detection_metadata:
                confidence = detection_metadata['confidence_score']
                confidence_factor = 1.0 - confidence  # Lower confidence = higher risk
                risk_factors.append(confidence_factor)
            
            return sum(risk_factors) / len(risk_factors) if risk_factors else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating defect risk score: {e}")
            return 0.0
    
    def _determine_defect_action(self, risk_score: float) -> str:
        """Determine required action based on defect risk score."""
        try:
            if risk_score >= 0.8:
                return 'immediate_stop'
            elif risk_score >= 0.6:
                return 'investigate'
            elif risk_score >= 0.4:
                return 'monitor'
            else:
                return 'continue'
        except Exception as e:
            logger.error(f"Error determining defect action: {e}")
            return 'unknown'
    
    def _calculate_overall_quality_score(self, quality_groups: Dict[str, List[Dict[str, Any]]]) -> float:
        """Calculate overall quality score from multiple quality assessments."""
        try:
            scores = []
            
            for quality_type, data_list in quality_groups.items():
                if data_list:
                    latest_data = max(data_list, key=lambda x: x.get('timestamp', ''))
                    score = latest_data.get('score', 0.0)
                    scores.append(score)
            
            return sum(scores) / len(scores) if scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating overall quality score: {e}")
            return 0.0
    
    def _determine_overall_quality_status(self, quality_score: float) -> str:
        """Determine overall quality status based on score."""
        try:
            if quality_score >= 0.9:
                return 'excellent'
            elif quality_score >= 0.7:
                return 'good'
            elif quality_score >= 0.5:
                return 'acceptable'
            elif quality_score >= 0.3:
                return 'poor'
            else:
                return 'critical'
        except Exception as e:
            logger.error(f"Error determining overall quality status: {e}")
            return 'unknown'
