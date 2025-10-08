"""
Spatially-Resolved Quality Analysis for PBF-LB/M Voxel Data

This module provides comprehensive spatially-resolved quality analysis capabilities
that enable detailed assessment of quality metrics at the voxel level. It integrates
with the multi-modal data fusion system to provide insights into quality variations
across the build volume.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from datetime import datetime
from scipy import ndimage
from scipy.spatial import distance_matrix
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

from src.core.domain.value_objects.voxel_coordinates import VoxelCoordinates
from src.core.domain.value_objects.quality_metrics import QualityMetrics
from src.core.domain.value_objects.defect_classification import DefectClassification

from ..core.multi_modal_fusion import FusedVoxelData

logger = logging.getLogger(__name__)


@dataclass
class SpatialQualityMetrics:
    """Spatial quality metrics for voxel analysis."""
    
    # Overall quality statistics
    mean_quality: float
    std_quality: float
    min_quality: float
    max_quality: float
    quality_percentiles: Dict[int, float]  # {25: value, 50: value, 75: value, 90: value, 95: value}
    
    # Spatial distribution metrics
    quality_gradient: np.ndarray  # 3D gradient of quality
    quality_variance: float
    spatial_autocorrelation: float
    quality_clusters: List[Dict]  # Quality cluster information
    
    # Defect analysis
    defect_density: float
    defect_distribution: Dict[str, float]  # Defect type distribution
    defect_clusters: List[Dict]  # Defect cluster information
    defect_correlation: Dict[str, float]  # Correlation with quality
    
    # Dimensional analysis
    dimensional_accuracy: Dict[str, float]  # X, Y, Z accuracy
    surface_roughness: Dict[str, float]  # Surface roughness metrics
    geometric_deviation: float
    
    # Process correlation
    process_quality_correlation: Dict[str, float]  # Correlation with process parameters
    layer_quality_trend: List[float]  # Quality trend across layers
    build_quality_trend: List[float]  # Quality trend across build time


@dataclass
class QualityRegion:
    """Quality region analysis result."""
    
    region_id: str
    region_type: str  # "high_quality", "low_quality", "defect_region", "transition_region"
    voxel_indices: List[Tuple[int, int, int]]
    centroid: Tuple[float, float, float]
    volume: float
    quality_metrics: Dict[str, float]
    defect_count: int
    dominant_defect_types: List[str]
    process_characteristics: Dict[str, float]


@dataclass
class QualityAnalysisConfig:
    """Configuration for spatial quality analysis."""
    
    # Quality thresholds
    high_quality_threshold: float = 90.0
    low_quality_threshold: float = 70.0
    defect_threshold: float = 0.05
    
    # Spatial analysis parameters
    spatial_resolution: float = 0.1  # mm
    cluster_min_size: int = 10  # minimum voxels in cluster
    gradient_smoothing: float = 1.0  # Gaussian smoothing sigma
    
    # Statistical parameters
    correlation_threshold: float = 0.3
    significance_level: float = 0.05
    bootstrap_samples: int = 1000
    
    # Visualization parameters
    color_map: str = "viridis"
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300


class SpatialQualityAnalyzer:
    """
    Spatially-resolved quality analyzer for PBF-LB/M voxel data.
    
    This class provides comprehensive quality analysis capabilities including:
    - Spatial quality distribution analysis
    - Defect clustering and characterization
    - Process-quality correlation analysis
    - Dimensional accuracy assessment
    - Quality trend analysis
    - Regional quality assessment
    """
    
    def __init__(self, config: QualityAnalysisConfig = None):
        """Initialize the spatial quality analyzer."""
        self.config = config or QualityAnalysisConfig()
        self.analysis_cache = {}
        
        logger.info("Spatial Quality Analyzer initialized")
    
    def analyze_spatial_quality(
        self,
        fused_data: Dict[Tuple[int, int, int], FusedVoxelData],
        voxel_grid_dimensions: Tuple[int, int, int],
        voxel_size: float
    ) -> SpatialQualityMetrics:
        """
        Perform comprehensive spatial quality analysis.
        
        Args:
            fused_data: Fused voxel data with quality information
            voxel_grid_dimensions: Dimensions of the voxel grid
            voxel_size: Size of each voxel in mm
            
        Returns:
            SpatialQualityMetrics: Comprehensive quality analysis results
        """
        try:
            logger.info("Starting spatial quality analysis...")
            
            # Extract quality data
            quality_data = self._extract_quality_data(fused_data, voxel_grid_dimensions)
            
            # Calculate overall quality statistics
            quality_stats = self._calculate_quality_statistics(quality_data)
            
            # Analyze spatial distribution
            spatial_distribution = self._analyze_spatial_distribution(quality_data, voxel_size)
            
            # Analyze defects
            defect_analysis = self._analyze_defects(fused_data, voxel_grid_dimensions)
            
            # Analyze dimensional accuracy
            dimensional_analysis = self._analyze_dimensional_accuracy(fused_data)
            
            # Analyze process correlations
            process_correlations = self._analyze_process_correlations(fused_data)
            
            # Analyze quality trends
            quality_trends = self._analyze_quality_trends(fused_data)
            
            # Combine all analyses
            spatial_metrics = SpatialQualityMetrics(
                mean_quality=quality_stats['mean'],
                std_quality=quality_stats['std'],
                min_quality=quality_stats['min'],
                max_quality=quality_stats['max'],
                quality_percentiles=quality_stats['percentiles'],
                quality_gradient=spatial_distribution['gradient'],
                quality_variance=spatial_distribution['variance'],
                spatial_autocorrelation=spatial_distribution['autocorrelation'],
                quality_clusters=spatial_distribution['clusters'],
                defect_density=defect_analysis['density'],
                defect_distribution=defect_analysis['distribution'],
                defect_clusters=defect_analysis['clusters'],
                defect_correlation=defect_analysis['correlation'],
                dimensional_accuracy=dimensional_analysis['accuracy'],
                surface_roughness=dimensional_analysis['roughness'],
                geometric_deviation=dimensional_analysis['deviation'],
                process_quality_correlation=process_correlations['correlations'],
                layer_quality_trend=quality_trends['layer_trend'],
                build_quality_trend=quality_trends['build_trend']
            )
            
            logger.info("Spatial quality analysis completed")
            return spatial_metrics
            
        except Exception as e:
            logger.error(f"Error in spatial quality analysis: {e}")
            raise
    
    def _extract_quality_data(
        self, 
        fused_data: Dict[Tuple[int, int, int], FusedVoxelData], 
        dimensions: Tuple[int, int, int]
    ) -> np.ndarray:
        """Extract quality data into a 3D array."""
        quality_array = np.full(dimensions, np.nan, dtype=np.float32)
        
        for voxel_idx, voxel_data in fused_data.items():
            if voxel_data.overall_quality_score is not None:
                quality_array[voxel_idx] = voxel_data.overall_quality_score
        
        return quality_array
    
    def _calculate_quality_statistics(self, quality_data: np.ndarray) -> Dict:
        """Calculate overall quality statistics."""
        # Remove NaN values
        valid_quality = quality_data[~np.isnan(quality_data)]
        
        if len(valid_quality) == 0:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'percentiles': {}
            }
        
        # Calculate basic statistics
        mean_quality = np.mean(valid_quality)
        std_quality = np.std(valid_quality)
        min_quality = np.min(valid_quality)
        max_quality = np.max(valid_quality)
        
        # Calculate percentiles
        percentiles = {
            25: np.percentile(valid_quality, 25),
            50: np.percentile(valid_quality, 50),
            75: np.percentile(valid_quality, 75),
            90: np.percentile(valid_quality, 90),
            95: np.percentile(valid_quality, 95)
        }
        
        return {
            'mean': mean_quality,
            'std': std_quality,
            'min': min_quality,
            'max': max_quality,
            'percentiles': percentiles
        }
    
    def _analyze_spatial_distribution(
        self, 
        quality_data: np.ndarray, 
        voxel_size: float
    ) -> Dict:
        """Analyze spatial distribution of quality."""
        # Calculate quality gradient
        gradient = self._calculate_quality_gradient(quality_data, voxel_size)
        
        # Calculate quality variance
        variance = np.nanvar(quality_data)
        
        # Calculate spatial autocorrelation
        autocorrelation = self._calculate_spatial_autocorrelation(quality_data)
        
        # Identify quality clusters
        clusters = self._identify_quality_clusters(quality_data)
        
        return {
            'gradient': gradient,
            'variance': variance,
            'autocorrelation': autocorrelation,
            'clusters': clusters
        }
    
    def _calculate_quality_gradient(self, quality_data: np.ndarray, voxel_size: float) -> np.ndarray:
        """Calculate 3D gradient of quality."""
        # Apply Gaussian smoothing
        smoothed_data = ndimage.gaussian_filter(quality_data, sigma=self.config.gradient_smoothing)
        
        # Calculate gradient
        gradient = np.gradient(smoothed_data, voxel_size)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(
            gradient[0]**2 + gradient[1]**2 + gradient[2]**2
        )
        
        return gradient_magnitude
    
    def _calculate_spatial_autocorrelation(self, quality_data: np.ndarray) -> float:
        """Calculate spatial autocorrelation of quality."""
        # Create distance matrix for a subset of voxels
        valid_indices = np.where(~np.isnan(quality_data))
        
        if len(valid_indices[0]) < 100:
            return 0.0
        
        # Sample a subset for computational efficiency
        sample_size = min(1000, len(valid_indices[0]))
        sample_indices = np.random.choice(len(valid_indices[0]), sample_size, replace=False)
        
        positions = np.column_stack([
            valid_indices[0][sample_indices],
            valid_indices[1][sample_indices],
            valid_indices[2][sample_indices]
        ])
        
        values = quality_data[valid_indices[0][sample_indices], 
                            valid_indices[1][sample_indices], 
                            valid_indices[2][sample_indices]]
        
        # Calculate distance matrix
        distances = distance_matrix(positions, positions)
        
        # Calculate autocorrelation
        autocorrelation = self._moran_i_autocorrelation(values, distances)
        
        return autocorrelation
    
    def _moran_i_autocorrelation(self, values: np.ndarray, distances: np.ndarray) -> float:
        """Calculate Moran's I autocorrelation coefficient."""
        n = len(values)
        mean_value = np.mean(values)
        
        # Create weight matrix (inverse distance weighting)
        weights = 1.0 / (distances + 1e-10)  # Add small value to avoid division by zero
        np.fill_diagonal(weights, 0)  # Remove self-connections
        
        # Normalize weights
        row_sums = weights.sum(axis=1)
        weights = weights / row_sums[:, np.newaxis]
        
        # Calculate Moran's I
        numerator = 0
        denominator = 0
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    numerator += weights[i, j] * (values[i] - mean_value) * (values[j] - mean_value)
            denominator += (values[i] - mean_value) ** 2
        
        if denominator == 0:
            return 0.0
        
        moran_i = (n / weights.sum()) * (numerator / denominator)
        
        return moran_i
    
    def _identify_quality_clusters(self, quality_data: np.ndarray) -> List[Dict]:
        """Identify quality clusters in the data."""
        clusters = []
        
        # Identify high-quality regions
        high_quality_mask = quality_data > self.config.high_quality_threshold
        high_quality_clusters = self._find_connected_components(high_quality_mask)
        
        for i, cluster in enumerate(high_quality_clusters):
            if len(cluster) >= self.config.cluster_min_size:
                clusters.append({
                    'cluster_id': f"high_quality_{i}",
                    'cluster_type': 'high_quality',
                    'voxel_count': len(cluster),
                    'mean_quality': np.mean([quality_data[idx] for idx in cluster]),
                    'voxel_indices': cluster
                })
        
        # Identify low-quality regions
        low_quality_mask = quality_data < self.config.low_quality_threshold
        low_quality_clusters = self._find_connected_components(low_quality_mask)
        
        for i, cluster in enumerate(low_quality_clusters):
            if len(cluster) >= self.config.cluster_min_size:
                clusters.append({
                    'cluster_id': f"low_quality_{i}",
                    'cluster_type': 'low_quality',
                    'voxel_count': len(cluster),
                    'mean_quality': np.mean([quality_data[idx] for idx in cluster]),
                    'voxel_indices': cluster
                })
        
        return clusters
    
    def _find_connected_components(self, mask: np.ndarray) -> List[List[Tuple[int, int, int]]]:
        """Find connected components in a 3D mask."""
        # Use scipy's label function for connected components
        labeled_array, num_features = ndimage.label(mask)
        
        clusters = []
        for i in range(1, num_features + 1):
            cluster_indices = np.where(labeled_array == i)
            cluster = list(zip(cluster_indices[0], cluster_indices[1], cluster_indices[2]))
            clusters.append(cluster)
        
        return clusters
    
    def _analyze_defects(
        self, 
        fused_data: Dict[Tuple[int, int, int], FusedVoxelData], 
        dimensions: Tuple[int, int, int]
    ) -> Dict:
        """Analyze defect distribution and characteristics."""
        # Extract defect data
        defect_data = []
        defect_types = {}
        
        for voxel_idx, voxel_data in fused_data.items():
            if voxel_data.defect_count > 0:
                defect_data.append({
                    'voxel_idx': voxel_idx,
                    'defect_count': voxel_data.defect_count,
                    'defect_types': voxel_data.defect_types or [],
                    'defect_probability': voxel_data.ct_defect_probability or 0.0
                })
                
                # Count defect types
                for defect_type in voxel_data.defect_types or []:
                    defect_types[defect_type] = defect_types.get(defect_type, 0) + 1
        
        # Calculate defect density
        total_voxels = np.prod(dimensions)
        defect_density = len(defect_data) / total_voxels if total_voxels > 0 else 0.0
        
        # Calculate defect distribution
        total_defects = sum(defect_types.values())
        defect_distribution = {
            defect_type: count / total_defects 
            for defect_type, count in defect_types.items()
        } if total_defects > 0 else {}
        
        # Identify defect clusters
        defect_clusters = self._identify_defect_clusters(defect_data, dimensions)
        
        # Calculate defect-quality correlation
        defect_correlation = self._calculate_defect_quality_correlation(fused_data)
        
        return {
            'density': defect_density,
            'distribution': defect_distribution,
            'clusters': defect_clusters,
            'correlation': defect_correlation
        }
    
    def _identify_defect_clusters(
        self, 
        defect_data: List[Dict], 
        dimensions: Tuple[int, int, int]
    ) -> List[Dict]:
        """Identify clusters of defects."""
        if not defect_data:
            return []
        
        # Create defect mask
        defect_mask = np.zeros(dimensions, dtype=bool)
        for defect in defect_data:
            defect_mask[defect['voxel_idx']] = True
        
        # Find connected components
        clusters = self._find_connected_components(defect_mask)
        
        # Convert to cluster information
        defect_clusters = []
        for i, cluster in enumerate(clusters):
            if len(cluster) >= self.config.cluster_min_size:
                cluster_defects = [d for d in defect_data if d['voxel_idx'] in cluster]
                
                defect_clusters.append({
                    'cluster_id': f"defect_cluster_{i}",
                    'voxel_count': len(cluster),
                    'defect_count': sum(d['defect_count'] for d in cluster_defects),
                    'defect_types': list(set(
                        defect_type for d in cluster_defects 
                        for defect_type in d['defect_types']
                    )),
                    'mean_defect_probability': np.mean([d['defect_probability'] for d in cluster_defects]),
                    'voxel_indices': cluster
                })
        
        return defect_clusters
    
    def _calculate_defect_quality_correlation(
        self, 
        fused_data: Dict[Tuple[int, int, int], FusedVoxelData]
    ) -> Dict[str, float]:
        """Calculate correlation between defects and quality."""
        correlations = {}
        
        # Extract data for correlation analysis
        quality_scores = []
        defect_counts = []
        defect_probabilities = []
        
        for voxel_data in fused_data.values():
            if (voxel_data.overall_quality_score is not None and 
                voxel_data.defect_count is not None):
                quality_scores.append(voxel_data.overall_quality_score)
                defect_counts.append(voxel_data.defect_count)
                defect_probabilities.append(voxel_data.ct_defect_probability or 0.0)
        
        if len(quality_scores) > 10:  # Minimum sample size
            # Correlation with defect count
            if len(set(defect_counts)) > 1:  # Check for variance
                corr, p_value = pearsonr(quality_scores, defect_counts)
                if p_value < self.config.significance_level:
                    correlations['defect_count'] = corr
            
            # Correlation with defect probability
            if len(set(defect_probabilities)) > 1:  # Check for variance
                corr, p_value = pearsonr(quality_scores, defect_probabilities)
                if p_value < self.config.significance_level:
                    correlations['defect_probability'] = corr
        
        return correlations
    
    def _analyze_dimensional_accuracy(self, fused_data: Dict[Tuple[int, int, int], FusedVoxelData]) -> Dict:
        """Analyze dimensional accuracy of the build."""
        # Extract dimensional accuracy data
        x_accuracies = []
        y_accuracies = []
        z_accuracies = []
        surface_roughnesses = []
        
        for voxel_data in fused_data.values():
            if voxel_data.dimensional_accuracy is not None:
                x_accuracies.append(voxel_data.dimensional_accuracy)
                y_accuracies.append(voxel_data.dimensional_accuracy)  # Placeholder
                z_accuracies.append(voxel_data.dimensional_accuracy)  # Placeholder
            
            if voxel_data.surface_roughness is not None:
                surface_roughnesses.append(voxel_data.surface_roughness)
        
        # Calculate accuracy metrics
        accuracy_metrics = {
            'x_accuracy': np.mean(x_accuracies) if x_accuracies else 0.0,
            'y_accuracy': np.mean(y_accuracies) if y_accuracies else 0.0,
            'z_accuracy': np.mean(z_accuracies) if z_accuracies else 0.0
        }
        
        # Calculate surface roughness metrics
        roughness_metrics = {
            'mean_roughness': np.mean(surface_roughnesses) if surface_roughnesses else 0.0,
            'std_roughness': np.std(surface_roughnesses) if surface_roughnesses else 0.0,
            'max_roughness': np.max(surface_roughnesses) if surface_roughnesses else 0.0
        }
        
        # Calculate geometric deviation
        geometric_deviation = np.std(list(accuracy_metrics.values()))
        
        return {
            'accuracy': accuracy_metrics,
            'roughness': roughness_metrics,
            'deviation': geometric_deviation
        }
    
    def _analyze_process_correlations(self, fused_data: Dict[Tuple[int, int, int], FusedVoxelData]) -> Dict:
        """Analyze correlations between process parameters and quality."""
        correlations = {}
        
        # Extract process and quality data
        quality_scores = []
        laser_powers = []
        scan_speeds = []
        temperatures = []
        
        for voxel_data in fused_data.values():
            if voxel_data.overall_quality_score is not None:
                quality_scores.append(voxel_data.overall_quality_score)
                laser_powers.append(voxel_data.laser_power)
                scan_speeds.append(voxel_data.scan_speed)
                temperatures.append(voxel_data.ispm_temperature or 0.0)
        
        if len(quality_scores) > 10:  # Minimum sample size
            # Correlation with laser power
            if len(set(laser_powers)) > 1:
                corr, p_value = pearsonr(quality_scores, laser_powers)
                if p_value < self.config.significance_level:
                    correlations['laser_power'] = corr
            
            # Correlation with scan speed
            if len(set(scan_speeds)) > 1:
                corr, p_value = pearsonr(quality_scores, scan_speeds)
                if p_value < self.config.significance_level:
                    correlations['scan_speed'] = corr
            
            # Correlation with temperature
            if len(set(temperatures)) > 1:
                corr, p_value = pearsonr(quality_scores, temperatures)
                if p_value < self.config.significance_level:
                    correlations['temperature'] = corr
        
        return {'correlations': correlations}
    
    def _analyze_quality_trends(self, fused_data: Dict[Tuple[int, int, int], FusedVoxelData]) -> Dict:
        """Analyze quality trends across layers and build time."""
        # Group by layer
        layer_quality = {}
        build_quality = []
        
        for voxel_data in fused_data.values():
            if (voxel_data.overall_quality_score is not None and 
                voxel_data.layer_number is not None):
                
                layer = voxel_data.layer_number
                if layer not in layer_quality:
                    layer_quality[layer] = []
                layer_quality[layer].append(voxel_data.overall_quality_score)
                
                build_quality.append(voxel_data.overall_quality_score)
        
        # Calculate layer trends
        layer_trend = []
        for layer in sorted(layer_quality.keys()):
            layer_trend.append(np.mean(layer_quality[layer]))
        
        # Calculate build trend (simplified - could be enhanced with time series analysis)
        build_trend = build_quality  # Placeholder
        
        return {
            'layer_trend': layer_trend,
            'build_trend': build_trend
        }
    
    def identify_quality_regions(
        self,
        fused_data: Dict[Tuple[int, int, int], FusedVoxelData],
        spatial_metrics: SpatialQualityMetrics
    ) -> List[QualityRegion]:
        """Identify distinct quality regions in the build."""
        regions = []
        
        # Process quality clusters
        for cluster in spatial_metrics.quality_clusters:
            region = QualityRegion(
                region_id=cluster['cluster_id'],
                region_type=cluster['cluster_type'],
                voxel_indices=cluster['voxel_indices'],
                centroid=self._calculate_region_centroid(cluster['voxel_indices']),
                volume=len(cluster['voxel_indices']) * 0.001,  # Convert to cm³
                quality_metrics={'mean_quality': cluster['mean_quality']},
                defect_count=0,  # Will be calculated
                dominant_defect_types=[],
                process_characteristics={}
            )
            regions.append(region)
        
        # Process defect clusters
        for cluster in spatial_metrics.defect_clusters:
            region = QualityRegion(
                region_id=cluster['cluster_id'],
                region_type='defect_region',
                voxel_indices=cluster['voxel_indices'],
                centroid=self._calculate_region_centroid(cluster['voxel_indices']),
                volume=len(cluster['voxel_indices']) * 0.001,  # Convert to cm³
                quality_metrics={'mean_defect_probability': cluster['mean_defect_probability']},
                defect_count=cluster['defect_count'],
                dominant_defect_types=cluster['defect_types'],
                process_characteristics={}
            )
            regions.append(region)
        
        return regions
    
    def _calculate_region_centroid(self, voxel_indices: List[Tuple[int, int, int]]) -> Tuple[float, float, float]:
        """Calculate centroid of a region."""
        if not voxel_indices:
            return (0.0, 0.0, 0.0)
        
        x_coords = [idx[0] for idx in voxel_indices]
        y_coords = [idx[1] for idx in voxel_indices]
        z_coords = [idx[2] for idx in voxel_indices]
        
        return (
            np.mean(x_coords),
            np.mean(y_coords),
            np.mean(z_coords)
        )
    
    def generate_quality_report(
        self,
        spatial_metrics: SpatialQualityMetrics,
        quality_regions: List[QualityRegion]
    ) -> str:
        """Generate a comprehensive quality analysis report."""
        report = []
        report.append("=== SPATIAL QUALITY ANALYSIS REPORT ===\n")
        
        # Overall quality statistics
        report.append("OVERALL QUALITY STATISTICS:")
        report.append(f"  Mean Quality: {spatial_metrics.mean_quality:.2f}")
        report.append(f"  Standard Deviation: {spatial_metrics.std_quality:.2f}")
        report.append(f"  Min Quality: {spatial_metrics.min_quality:.2f}")
        report.append(f"  Max Quality: {spatial_metrics.max_quality:.2f}")
        report.append(f"  Quality Variance: {spatial_metrics.quality_variance:.4f}")
        report.append(f"  Spatial Autocorrelation: {spatial_metrics.spatial_autocorrelation:.4f}\n")
        
        # Quality percentiles
        report.append("QUALITY PERCENTILES:")
        for percentile, value in spatial_metrics.quality_percentiles.items():
            report.append(f"  {percentile}th percentile: {value:.2f}")
        report.append("")
        
        # Defect analysis
        report.append("DEFECT ANALYSIS:")
        report.append(f"  Defect Density: {spatial_metrics.defect_density:.4f}")
        report.append("  Defect Distribution:")
        for defect_type, percentage in spatial_metrics.defect_distribution.items():
            report.append(f"    {defect_type}: {percentage:.2%}")
        report.append("  Defect-Quality Correlations:")
        for correlation_type, value in spatial_metrics.defect_correlation.items():
            report.append(f"    {correlation_type}: {value:.4f}")
        report.append("")
        
        # Dimensional analysis
        report.append("DIMENSIONAL ANALYSIS:")
        report.append(f"  Geometric Deviation: {spatial_metrics.geometric_deviation:.4f}")
        report.append("  Dimensional Accuracy:")
        for axis, accuracy in spatial_metrics.dimensional_accuracy.items():
            report.append(f"    {axis}: {accuracy:.4f}")
        report.append("  Surface Roughness:")
        for metric, value in spatial_metrics.surface_roughness.items():
            report.append(f"    {metric}: {value:.4f}")
        report.append("")
        
        # Process correlations
        report.append("PROCESS-QUALITY CORRELATIONS:")
        for param, correlation in spatial_metrics.process_quality_correlation.items():
            report.append(f"  {param}: {correlation:.4f}")
        report.append("")
        
        # Quality regions
        report.append("QUALITY REGIONS:")
        report.append(f"  Total Regions Identified: {len(quality_regions)}")
        for region in quality_regions:
            report.append(f"  Region {region.region_id}:")
            report.append(f"    Type: {region.region_type}")
            report.append(f"    Volume: {region.volume:.4f} cm³")
            report.append(f"    Voxel Count: {len(region.voxel_indices)}")
            report.append(f"    Defect Count: {region.defect_count}")
            if region.dominant_defect_types:
                report.append(f"    Dominant Defects: {', '.join(region.dominant_defect_types)}")
        report.append("")
        
        # Quality trends
        report.append("QUALITY TRENDS:")
        report.append(f"  Layer Quality Trend: {len(spatial_metrics.layer_quality_trend)} layers analyzed")
        if spatial_metrics.layer_quality_trend:
            trend_direction = "improving" if spatial_metrics.layer_quality_trend[-1] > spatial_metrics.layer_quality_trend[0] else "declining"
            report.append(f"  Overall Trend: {trend_direction}")
        report.append("")
        
        report.append("=== END OF REPORT ===")
        
        return "\n".join(report)
    
    def export_analysis_results(
        self,
        spatial_metrics: SpatialQualityMetrics,
        quality_regions: List[QualityRegion],
        output_path: str
    ):
        """Export analysis results to file."""
        try:
            import json
            from pathlib import Path
            
            # Create output directory if it doesn't exist
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare data for export
            export_data = {
                'spatial_metrics': {
                    'mean_quality': spatial_metrics.mean_quality,
                    'std_quality': spatial_metrics.std_quality,
                    'min_quality': spatial_metrics.min_quality,
                    'max_quality': spatial_metrics.max_quality,
                    'quality_percentiles': spatial_metrics.quality_percentiles,
                    'quality_variance': spatial_metrics.quality_variance,
                    'spatial_autocorrelation': spatial_metrics.spatial_autocorrelation,
                    'defect_density': spatial_metrics.defect_density,
                    'defect_distribution': spatial_metrics.defect_distribution,
                    'dimensional_accuracy': spatial_metrics.dimensional_accuracy,
                    'surface_roughness': spatial_metrics.surface_roughness,
                    'geometric_deviation': spatial_metrics.geometric_deviation,
                    'process_quality_correlation': spatial_metrics.process_quality_correlation
                },
                'quality_regions': [
                    {
                        'region_id': region.region_id,
                        'region_type': region.region_type,
                        'centroid': region.centroid,
                        'volume': region.volume,
                        'voxel_count': len(region.voxel_indices),
                        'defect_count': region.defect_count,
                        'dominant_defect_types': region.dominant_defect_types,
                        'quality_metrics': region.quality_metrics
                    }
                    for region in quality_regions
                ],
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Export to JSON
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Spatial quality analysis results exported to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting analysis results: {e}")
            raise
