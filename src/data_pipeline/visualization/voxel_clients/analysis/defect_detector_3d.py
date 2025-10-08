"""
3D Defect Detection for PBF-LB/M Voxel Analysis

This module provides advanced 3D defect detection algorithms specifically designed
for PBF-LB/M (Powder Bed Fusion - Laser Beam/Metal) additive manufacturing research.
It identifies and classifies various types of defects in voxel data.
"""

import numpy as np
from scipy import ndimage
from scipy.spatial import distance_matrix
from sklearn.cluster import DBSCAN
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from datetime import datetime
from enum import Enum

from src.core.domain.value_objects.defect_classification import DefectClassification
from src.core.domain.value_objects.voxel_coordinates import VoxelCoordinates
from ..core.multi_modal_fusion import FusedVoxelData

logger = logging.getLogger(__name__)


class DefectType(Enum):
    """Types of defects that can be detected."""
    POROSITY = "porosity"
    CRACK = "crack"
    DELAMINATION = "delamination"
    INCLUSION = "inclusion"
    SURFACE_ROUGHNESS = "surface_roughness"
    DIMENSIONAL_DEVIATION = "dimensional_deviation"
    UNFUSED_POWDER = "unfused_powder"
    KEYHOLE = "keyhole"
    BALLING = "balling"
    LACK_OF_FUSION = "lack_of_fusion"


@dataclass
class DefectDetectionConfig:
    """Configuration for 3D defect detection."""
    
    # Detection thresholds
    porosity_threshold: float = 0.05  # 5% porosity threshold
    crack_threshold: float = 0.1  # Crack detection threshold
    delamination_threshold: float = 0.2  # Delamination threshold
    inclusion_threshold: float = 0.3  # Inclusion detection threshold
    
    # Spatial parameters
    min_defect_size: int = 5  # Minimum voxels in defect
    max_defect_size: int = 10000  # Maximum voxels in defect
    connectivity: int = 26  # 3D connectivity (6, 18, or 26)
    
    # Clustering parameters
    cluster_eps: float = 2.0  # DBSCAN epsilon
    cluster_min_samples: int = 5  # DBSCAN minimum samples
    
    # Quality parameters
    quality_threshold: float = 80.0  # Quality score threshold
    confidence_threshold: float = 0.7  # Detection confidence threshold
    
    # Performance parameters
    enable_parallel_processing: bool = True
    max_voxels_per_batch: int = 100000
    memory_limit_gb: float = 4.0


@dataclass
class DefectDetectionResult:
    """Result of defect detection operation."""
    
    success: bool
    defects: List[DefectClassification]
    defect_statistics: Dict[str, Any]
    detection_time: float
    voxel_count: int
    defect_count: int
    error_message: Optional[str] = None


class DefectDetector3D:
    """
    3D defect detector for PBF-LB/M voxel analysis.
    
    This class provides comprehensive defect detection capabilities including:
    - Porosity detection and analysis
    - Crack detection and characterization
    - Delamination identification
    - Inclusion detection
    - Surface defect analysis
    - Dimensional deviation detection
    - Defect clustering and classification
    """
    
    def __init__(self, config: DefectDetectionConfig = None):
        """Initialize the 3D defect detector."""
        self.config = config or DefectDetectionConfig()
        self.detection_cache = {}
        
        logger.info("3D Defect Detector initialized")
    
    def detect_defects(
        self,
        fused_data: Dict[Tuple[int, int, int], FusedVoxelData],
        voxel_grid_dimensions: Tuple[int, int, int],
        voxel_size: float
    ) -> DefectDetectionResult:
        """
        Detect defects in fused voxel data.
        
        Args:
            fused_data: Fused voxel data with quality information
            voxel_grid_dimensions: Dimensions of the voxel grid
            voxel_size: Size of each voxel in mm
            
        Returns:
            DefectDetectionResult: Detection results with classified defects
        """
        try:
            start_time = datetime.now()
            
            # Extract defect data
            defect_data = self._extract_defect_data(fused_data, voxel_grid_dimensions)
            
            # Detect different types of defects
            all_defects = []
            
            # Porosity detection
            porosity_defects = self._detect_porosity(defect_data, voxel_size)
            all_defects.extend(porosity_defects)
            
            # Crack detection
            crack_defects = self._detect_cracks(defect_data, voxel_size)
            all_defects.extend(crack_defects)
            
            # Delamination detection
            delamination_defects = self._detect_delamination(defect_data, voxel_size)
            all_defects.extend(delamination_defects)
            
            # Inclusion detection
            inclusion_defects = self._detect_inclusions(defect_data, voxel_size)
            all_defects.extend(inclusion_defects)
            
            # Surface defect detection
            surface_defects = self._detect_surface_defects(defect_data, voxel_size)
            all_defects.extend(surface_defects)
            
            # Dimensional deviation detection
            dimensional_defects = self._detect_dimensional_deviations(defect_data, voxel_size)
            all_defects.extend(dimensional_defects)
            
            # Filter and validate defects
            validated_defects = self._validate_defects(all_defects)
            
            # Calculate statistics
            defect_statistics = self._calculate_defect_statistics(validated_defects)
            
            # Calculate detection time
            detection_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = DefectDetectionResult(
                success=True,
                defects=validated_defects,
                defect_statistics=defect_statistics,
                detection_time=detection_time,
                voxel_count=len(fused_data),
                defect_count=len(validated_defects)
            )
            
            logger.info(f"Defect detection completed: {len(validated_defects)} defects found in {detection_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in defect detection: {e}")
            return DefectDetectionResult(
                success=False,
                defects=[],
                defect_statistics={},
                detection_time=0.0,
                voxel_count=0,
                defect_count=0,
                error_message=str(e)
            )
    
    def _extract_defect_data(
        self, 
        fused_data: Dict[Tuple[int, int, int], FusedVoxelData], 
        dimensions: Tuple[int, int, int]
    ) -> Dict[str, np.ndarray]:
        """Extract defect-related data from fused voxel data."""
        # Initialize arrays
        porosity_map = np.zeros(dimensions, dtype=np.float32)
        quality_map = np.zeros(dimensions, dtype=np.float32)
        density_map = np.zeros(dimensions, dtype=np.float32)
        temperature_map = np.zeros(dimensions, dtype=np.float32)
        voxel_mask = np.zeros(dimensions, dtype=bool)
        
        # Fill arrays with data
        for voxel_idx, voxel_data in fused_data.items():
            if self._is_valid_voxel_index(voxel_idx, dimensions):
                porosity_map[voxel_idx] = voxel_data.ct_porosity or 0.0
                quality_map[voxel_idx] = voxel_data.overall_quality_score or 100.0
                density_map[voxel_idx] = voxel_data.ct_density or 4.43  # Ti-6Al-4V density
                temperature_map[voxel_idx] = voxel_data.ispm_temperature or 0.0
                voxel_mask[voxel_idx] = True
        
        return {
            'porosity': porosity_map,
            'quality': quality_map,
            'density': density_map,
            'temperature': temperature_map,
            'voxel_mask': voxel_mask
        }
    
    def _detect_porosity(self, defect_data: Dict[str, np.ndarray], voxel_size: float) -> List[DefectClassification]:
        """Detect porosity defects."""
        porosity_defects = []
        porosity_map = defect_data['porosity']
        voxel_mask = defect_data['voxel_mask']
        
        # Find high porosity regions
        high_porosity_mask = (porosity_map > self.config.porosity_threshold) & voxel_mask
        
        if not np.any(high_porosity_mask):
            return porosity_defects
        
        # Find connected components
        labeled_porosity, num_features = ndimage.label(high_porosity_mask, structure=ndimage.generate_binary_structure(3, self.config.connectivity))
        
        for i in range(1, num_features + 1):
            porosity_region = (labeled_porosity == i)
            porosity_voxels = np.where(porosity_region)
            
            if len(porosity_voxels[0]) < self.config.min_defect_size:
                continue
            
            # Calculate porosity statistics
            porosity_values = porosity_map[porosity_region]
            avg_porosity = np.mean(porosity_values)
            max_porosity = np.max(porosity_values)
            
            # Calculate centroid
            centroid = self._calculate_centroid(porosity_voxels, voxel_size)
            
            # Calculate volume
            volume = len(porosity_voxels[0]) * (voxel_size ** 3)
            
            # Determine severity
            severity = self._determine_porosity_severity(avg_porosity)
            
            # Create defect classification
            defect = DefectClassification(
                defect_id=f"porosity_{len(porosity_defects)}",
                defect_type=DefectType.POROSITY.value,
                severity=severity,
                location="internal",
                size=volume,
                x_coordinate=centroid[0],
                y_coordinate=centroid[1],
                z_coordinate=centroid[2],
                porosity=avg_porosity,
                detection_method="ct_scan",
                detection_confidence=min(1.0, avg_porosity / self.config.porosity_threshold)
            )
            
            porosity_defects.append(defect)
        
        return porosity_defects
    
    def _detect_cracks(self, defect_data: Dict[str, np.ndarray], voxel_size: float) -> List[DefectClassification]:
        """Detect crack defects."""
        crack_defects = []
        quality_map = defect_data['quality']
        voxel_mask = defect_data['voxel_mask']
        
        # Find low quality regions (potential cracks)
        low_quality_mask = (quality_map < self.config.quality_threshold) & voxel_mask
        
        if not np.any(low_quality_mask):
            return crack_defects
        
        # Apply morphological operations to detect crack-like structures
        # Erosion to thin the regions
        eroded = ndimage.binary_erosion(low_quality_mask, structure=ndimage.generate_binary_structure(3, 1))
        
        # Find connected components
        labeled_cracks, num_features = ndimage.label(eroded, structure=ndimage.generate_binary_structure(3, 1))
        
        for i in range(1, num_features + 1):
            crack_region = (labeled_cracks == i)
            crack_voxels = np.where(crack_region)
            
            if len(crack_voxels[0]) < self.config.min_defect_size:
                continue
            
            # Calculate crack characteristics
            crack_length = self._calculate_crack_length(crack_voxels, voxel_size)
            crack_width = self._calculate_crack_width(crack_voxels, voxel_size)
            
            # Calculate centroid
            centroid = self._calculate_centroid(crack_voxels, voxel_size)
            
            # Determine severity based on length and width
            severity = self._determine_crack_severity(crack_length, crack_width)
            
            # Create defect classification
            defect = DefectClassification(
                defect_id=f"crack_{len(crack_defects)}",
                defect_type=DefectType.CRACK.value,
                severity=severity,
                location="internal",
                size=crack_length * crack_width * voxel_size,
                x_coordinate=centroid[0],
                y_coordinate=centroid[1],
                z_coordinate=centroid[2],
                detection_method="quality_analysis",
                detection_confidence=0.8  # High confidence for crack detection
            )
            
            crack_defects.append(defect)
        
        return crack_defects
    
    def _detect_delamination(self, defect_data: Dict[str, np.ndarray], voxel_size: float) -> List[DefectClassification]:
        """Detect delamination defects."""
        delamination_defects = []
        quality_map = defect_data['quality']
        voxel_mask = defect_data['voxel_mask']
        
        # Delamination typically occurs at layer boundaries
        # Look for low quality regions that span multiple layers
        low_quality_mask = (quality_map < self.config.quality_threshold * 0.8) & voxel_mask
        
        if not np.any(low_quality_mask):
            return delamination_defects
        
        # Find connected components
        labeled_delamination, num_features = ndimage.label(low_quality_mask, structure=ndimage.generate_binary_structure(3, self.config.connectivity))
        
        for i in range(1, num_features + 1):
            delamination_region = (labeled_delamination == i)
            delamination_voxels = np.where(delamination_region)
            
            if len(delamination_voxels[0]) < self.config.min_defect_size:
                continue
            
            # Check if defect spans multiple layers (delamination characteristic)
            z_coords = delamination_voxels[2]
            z_range = np.max(z_coords) - np.min(z_coords)
            
            if z_range < 2:  # Must span at least 2 layers
                continue
            
            # Calculate delamination area
            area = self._calculate_delamination_area(delamination_voxels, voxel_size)
            
            # Calculate centroid
            centroid = self._calculate_centroid(delamination_voxels, voxel_size)
            
            # Determine severity
            severity = self._determine_delamination_severity(area, z_range)
            
            # Create defect classification
            defect = DefectClassification(
                defect_id=f"delamination_{len(delamination_defects)}",
                defect_type=DefectType.DELAMINATION.value,
                severity=severity,
                location="layer_boundary",
                size=area * z_range * voxel_size,
                x_coordinate=centroid[0],
                y_coordinate=centroid[1],
                z_coordinate=centroid[2],
                detection_method="layer_analysis",
                detection_confidence=0.7
            )
            
            delamination_defects.append(defect)
        
        return delamination_defects
    
    def _detect_inclusions(self, defect_data: Dict[str, np.ndarray], voxel_size: float) -> List[DefectClassification]:
        """Detect inclusion defects."""
        inclusion_defects = []
        density_map = defect_data['density']
        voxel_mask = defect_data['voxel_mask']
        
        # Inclusions typically have different density than the base material
        # Look for density variations
        density_std = np.std(density_map[voxel_mask])
        density_mean = np.mean(density_map[voxel_mask])
        
        # Find regions with significantly different density
        inclusion_mask = (np.abs(density_map - density_mean) > 2 * density_std) & voxel_mask
        
        if not np.any(inclusion_mask):
            return inclusion_defects
        
        # Find connected components
        labeled_inclusions, num_features = ndimage.label(inclusion_mask, structure=ndimage.generate_binary_structure(3, self.config.connectivity))
        
        for i in range(1, num_features + 1):
            inclusion_region = (labeled_inclusions == i)
            inclusion_voxels = np.where(inclusion_region)
            
            if len(inclusion_voxels[0]) < self.config.min_defect_size:
                continue
            
            # Calculate inclusion characteristics
            inclusion_density = np.mean(density_map[inclusion_region])
            density_deviation = abs(inclusion_density - density_mean) / density_mean
            
            # Calculate centroid
            centroid = self._calculate_centroid(inclusion_voxels, voxel_size)
            
            # Calculate volume
            volume = len(inclusion_voxels[0]) * (voxel_size ** 3)
            
            # Determine severity
            severity = self._determine_inclusion_severity(density_deviation, volume)
            
            # Create defect classification
            defect = DefectClassification(
                defect_id=f"inclusion_{len(inclusion_defects)}",
                defect_type=DefectType.INCLUSION.value,
                severity=severity,
                location="internal",
                size=volume,
                x_coordinate=centroid[0],
                y_coordinate=centroid[1],
                z_coordinate=centroid[2],
                detection_method="density_analysis",
                detection_confidence=min(1.0, density_deviation)
            )
            
            inclusion_defects.append(defect)
        
        return inclusion_defects
    
    def _detect_surface_defects(self, defect_data: Dict[str, np.ndarray], voxel_size: float) -> List[DefectClassification]:
        """Detect surface defects."""
        surface_defects = []
        quality_map = defect_data['quality']
        voxel_mask = defect_data['voxel_mask']
        
        # Surface defects are typically at the boundaries of the voxel mask
        # Find surface voxels
        surface_mask = self._find_surface_voxels(voxel_mask)
        
        # Find low quality surface regions
        surface_defect_mask = (quality_map < self.config.quality_threshold * 0.9) & surface_mask
        
        if not np.any(surface_defect_mask):
            return surface_defects
        
        # Find connected components
        labeled_surface, num_features = ndimage.label(surface_defect_mask, structure=ndimage.generate_binary_structure(3, self.config.connectivity))
        
        for i in range(1, num_features + 1):
            surface_region = (labeled_surface == i)
            surface_voxels = np.where(surface_region)
            
            if len(surface_voxels[0]) < self.config.min_defect_size:
                continue
            
            # Calculate surface area
            surface_area = self._calculate_surface_area(surface_voxels, voxel_size)
            
            # Calculate centroid
            centroid = self._calculate_centroid(surface_voxels, voxel_size)
            
            # Determine severity
            severity = self._determine_surface_defect_severity(surface_area)
            
            # Create defect classification
            defect = DefectClassification(
                defect_id=f"surface_{len(surface_defects)}",
                defect_type=DefectType.SURFACE_ROUGHNESS.value,
                severity=severity,
                location="surface",
                size=surface_area,
                x_coordinate=centroid[0],
                y_coordinate=centroid[1],
                z_coordinate=centroid[2],
                detection_method="surface_analysis",
                detection_confidence=0.6
            )
            
            surface_defects.append(defect)
        
        return surface_defects
    
    def _detect_dimensional_deviations(self, defect_data: Dict[str, np.ndarray], voxel_size: float) -> List[DefectClassification]:
        """Detect dimensional deviation defects."""
        dimensional_defects = []
        quality_map = defect_data['quality']
        voxel_mask = defect_data['voxel_mask']
        
        # Dimensional deviations are typically detected by comparing with CAD model
        # For now, we'll use quality variations as a proxy
        quality_std = np.std(quality_map[voxel_mask])
        quality_mean = np.mean(quality_map[voxel_mask])
        
        # Find regions with significant quality variations
        deviation_mask = (np.abs(quality_map - quality_mean) > 2 * quality_std) & voxel_mask
        
        if not np.any(deviation_mask):
            return dimensional_defects
        
        # Find connected components
        labeled_deviations, num_features = ndimage.label(deviation_mask, structure=ndimage.generate_binary_structure(3, self.config.connectivity))
        
        for i in range(1, num_features + 1):
            deviation_region = (labeled_deviations == i)
            deviation_voxels = np.where(deviation_region)
            
            if len(deviation_voxels[0]) < self.config.min_defect_size:
                continue
            
            # Calculate deviation characteristics
            deviation_magnitude = np.mean(np.abs(quality_map[deviation_region] - quality_mean))
            
            # Calculate centroid
            centroid = self._calculate_centroid(deviation_voxels, voxel_size)
            
            # Calculate volume
            volume = len(deviation_voxels[0]) * (voxel_size ** 3)
            
            # Determine severity
            severity = self._determine_dimensional_severity(deviation_magnitude)
            
            # Create defect classification
            defect = DefectClassification(
                defect_id=f"dimensional_{len(dimensional_defects)}",
                defect_type=DefectType.DIMENSIONAL_DEVIATION.value,
                severity=severity,
                location="internal",
                size=volume,
                x_coordinate=centroid[0],
                y_coordinate=centroid[1],
                z_coordinate=centroid[2],
                detection_method="dimensional_analysis",
                detection_confidence=min(1.0, deviation_magnitude / quality_std)
            )
            
            dimensional_defects.append(defect)
        
        return dimensional_defects
    
    def _validate_defects(self, defects: List[DefectClassification]) -> List[DefectClassification]:
        """Validate and filter defects."""
        validated_defects = []
        
        for defect in defects:
            # Check size constraints
            if defect.size < self.config.min_defect_size * (0.1 ** 3):  # Convert to mm³
                continue
            
            if defect.size > self.config.max_defect_size * (0.1 ** 3):
                continue
            
            # Check confidence threshold
            if defect.detection_confidence < self.config.confidence_threshold:
                continue
            
            validated_defects.append(defect)
        
        return validated_defects
    
    def _calculate_defect_statistics(self, defects: List[DefectClassification]) -> Dict[str, Any]:
        """Calculate defect statistics."""
        if not defects:
            return {
                'total_defects': 0,
                'defect_types': {},
                'severity_distribution': {},
                'total_defect_volume': 0.0,
                'average_defect_size': 0.0
            }
        
        # Count by type
        defect_types = {}
        for defect in defects:
            defect_types[defect.defect_type] = defect_types.get(defect.defect_type, 0) + 1
        
        # Count by severity
        severity_distribution = {}
        for defect in defects:
            severity_distribution[defect.severity] = severity_distribution.get(defect.severity, 0) + 1
        
        # Calculate volumes
        total_volume = sum(defect.size for defect in defects)
        average_size = total_volume / len(defects)
        
        return {
            'total_defects': len(defects),
            'defect_types': defect_types,
            'severity_distribution': severity_distribution,
            'total_defect_volume': total_volume,
            'average_defect_size': average_size,
            'defect_density': len(defects) / total_volume if total_volume > 0 else 0
        }
    
    # Helper methods
    def _is_valid_voxel_index(self, voxel_idx: Tuple[int, int, int], dimensions: Tuple[int, int, int]) -> bool:
        """Check if voxel index is valid."""
        return (0 <= voxel_idx[0] < dimensions[0] and 
                0 <= voxel_idx[1] < dimensions[1] and 
                0 <= voxel_idx[2] < dimensions[2])
    
    def _calculate_centroid(self, voxel_coords: Tuple[np.ndarray, np.ndarray, np.ndarray], voxel_size: float) -> Tuple[float, float, float]:
        """Calculate centroid of voxel coordinates."""
        x_coords = voxel_coords[0] * voxel_size
        y_coords = voxel_coords[1] * voxel_size
        z_coords = voxel_coords[2] * voxel_size
        
        return (np.mean(x_coords), np.mean(y_coords), np.mean(z_coords))
    
    def _calculate_crack_length(self, voxel_coords: Tuple[np.ndarray, np.ndarray, np.ndarray], voxel_size: float) -> float:
        """Calculate crack length."""
        # Simple approximation: maximum distance between any two points
        coords = np.column_stack([voxel_coords[0], voxel_coords[1], voxel_coords[2]])
        distances = distance_matrix(coords, coords)
        max_distance = np.max(distances)
        return max_distance * voxel_size
    
    def _calculate_crack_width(self, voxel_coords: Tuple[np.ndarray, np.ndarray, np.ndarray], voxel_size: float) -> float:
        """Calculate crack width."""
        # Simple approximation: average of x and y ranges
        x_range = np.max(voxel_coords[0]) - np.min(voxel_coords[0])
        y_range = np.max(voxel_coords[1]) - np.min(voxel_coords[1])
        return (x_range + y_range) / 2 * voxel_size
    
    def _calculate_delamination_area(self, voxel_coords: Tuple[np.ndarray, np.ndarray, np.ndarray], voxel_size: float) -> float:
        """Calculate delamination area."""
        # Project to 2D and calculate area
        x_range = np.max(voxel_coords[0]) - np.min(voxel_coords[0])
        y_range = np.max(voxel_coords[1]) - np.min(voxel_coords[1])
        return x_range * y_range * (voxel_size ** 2)
    
    def _calculate_surface_area(self, voxel_coords: Tuple[np.ndarray, np.ndarray, np.ndarray], voxel_size: float) -> float:
        """Calculate surface area."""
        # Simple approximation: number of voxels * voxel face area
        return len(voxel_coords[0]) * 6 * (voxel_size ** 2)
    
    def _find_surface_voxels(self, voxel_mask: np.ndarray) -> np.ndarray:
        """Find surface voxels."""
        # Use morphological operations to find surface
        eroded = ndimage.binary_erosion(voxel_mask, structure=ndimage.generate_binary_structure(3, 1))
        surface_mask = voxel_mask & ~eroded
        return surface_mask
    
    # Severity determination methods
    def _determine_porosity_severity(self, porosity: float) -> str:
        """Determine porosity severity."""
        if porosity < 0.1:
            return "low"
        elif porosity < 0.2:
            return "medium"
        else:
            return "high"
    
    def _determine_crack_severity(self, length: float, width: float) -> str:
        """Determine crack severity."""
        if length < 1.0 and width < 0.1:
            return "low"
        elif length < 5.0 and width < 0.5:
            return "medium"
        else:
            return "high"
    
    def _determine_delamination_severity(self, area: float, z_range: float) -> str:
        """Determine delamination severity."""
        if area < 1.0 and z_range < 2:
            return "low"
        elif area < 10.0 and z_range < 5:
            return "medium"
        else:
            return "high"
    
    def _determine_inclusion_severity(self, density_deviation: float, volume: float) -> str:
        """Determine inclusion severity."""
        if density_deviation < 0.1 and volume < 0.1:
            return "low"
        elif density_deviation < 0.3 and volume < 1.0:
            return "medium"
        else:
            return "high"
    
    def _determine_surface_defect_severity(self, surface_area: float) -> str:
        """Determine surface defect severity."""
        if surface_area < 1.0:
            return "low"
        elif surface_area < 10.0:
            return "medium"
        else:
            return "high"
    
    def _determine_dimensional_severity(self, deviation_magnitude: float) -> str:
        """Determine dimensional deviation severity."""
        if deviation_magnitude < 5.0:
            return "low"
        elif deviation_magnitude < 15.0:
            return "medium"
        else:
            return "high"
