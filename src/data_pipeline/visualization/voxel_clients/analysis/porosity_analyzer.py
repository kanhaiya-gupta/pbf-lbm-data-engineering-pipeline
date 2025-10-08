"""
Porosity Analysis for PBF-LB/M Voxel Data

This module provides comprehensive porosity analysis capabilities for PBF-LB/M
(Powder Bed Fusion - Laser Beam/Metal) additive manufacturing research. It
analyzes porosity distribution, characteristics, and correlations with process parameters.
"""

import numpy as np
from scipy import ndimage
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import DBSCAN
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from src.core.domain.value_objects.voxel_coordinates import VoxelCoordinates
from ..core.multi_modal_fusion import FusedVoxelData

logger = logging.getLogger(__name__)


@dataclass
class PorosityAnalysisConfig:
    """Configuration for porosity analysis."""
    
    # Porosity thresholds
    porosity_threshold: float = 0.05  # 5% porosity threshold
    high_porosity_threshold: float = 0.15  # 15% high porosity threshold
    critical_porosity_threshold: float = 0.25  # 25% critical porosity threshold
    
    # Analysis parameters
    min_porosity_cluster_size: int = 10  # Minimum voxels in porosity cluster
    max_porosity_cluster_size: int = 10000  # Maximum voxels in porosity cluster
    connectivity: int = 26  # 3D connectivity (6, 18, or 26)
    
    # Statistical parameters
    correlation_threshold: float = 0.3
    significance_level: float = 0.05
    bootstrap_samples: int = 1000
    
    # Visualization parameters
    color_map: str = "Reds"  # Color map for porosity visualization
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300


@dataclass
class PorosityCluster:
    """Porosity cluster information."""
    
    cluster_id: str
    voxel_indices: List[Tuple[int, int, int]]
    centroid: Tuple[float, float, float]
    volume: float  # mm³
    porosity: float  # Average porosity in cluster
    max_porosity: float  # Maximum porosity in cluster
    min_porosity: float  # Minimum porosity in cluster
    porosity_std: float  # Standard deviation of porosity
    sphericity: float  # Sphericity of the cluster
    aspect_ratio: float  # Aspect ratio of the cluster
    surface_area: float  # Surface area in mm²
    equivalent_diameter: float  # Equivalent spherical diameter in mm


@dataclass
class PorosityAnalysisResult:
    """Result of porosity analysis operation."""
    
    success: bool
    porosity_statistics: Dict[str, Any]
    porosity_clusters: List[PorosityCluster]
    spatial_distribution: Dict[str, Any]
    process_correlations: Dict[str, float]
    analysis_time: float
    voxel_count: int
    error_message: Optional[str] = None


class PorosityAnalyzer:
    """
    Porosity analyzer for PBF-LB/M voxel data.
    
    This class provides comprehensive porosity analysis capabilities including:
    - Porosity distribution analysis
    - Porosity clustering and characterization
    - Spatial distribution analysis
    - Process parameter correlations
    - Porosity trend analysis
    - Quality impact assessment
    """
    
    def __init__(self, config: PorosityAnalysisConfig = None):
        """Initialize the porosity analyzer."""
        self.config = config or PorosityAnalysisConfig()
        self.analysis_cache = {}
        
        logger.info("Porosity Analyzer initialized")
    
    def analyze_porosity(
        self,
        fused_data: Dict[Tuple[int, int, int], FusedVoxelData],
        voxel_grid_dimensions: Tuple[int, int, int],
        voxel_size: float
    ) -> PorosityAnalysisResult:
        """
        Perform comprehensive porosity analysis.
        
        Args:
            fused_data: Fused voxel data with porosity information
            voxel_grid_dimensions: Dimensions of the voxel grid
            voxel_size: Size of each voxel in mm
            
        Returns:
            PorosityAnalysisResult: Comprehensive porosity analysis results
        """
        try:
            start_time = datetime.now()
            
            # Extract porosity data
            porosity_data = self._extract_porosity_data(fused_data, voxel_grid_dimensions)
            
            # Calculate porosity statistics
            porosity_stats = self._calculate_porosity_statistics(porosity_data)
            
            # Analyze porosity clusters
            porosity_clusters = self._analyze_porosity_clusters(porosity_data, voxel_size)
            
            # Analyze spatial distribution
            spatial_distribution = self._analyze_spatial_distribution(porosity_data, voxel_size)
            
            # Analyze process correlations
            process_correlations = self._analyze_process_correlations(fused_data, porosity_data)
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = PorosityAnalysisResult(
                success=True,
                porosity_statistics=porosity_stats,
                porosity_clusters=porosity_clusters,
                spatial_distribution=spatial_distribution,
                process_correlations=process_correlations,
                analysis_time=analysis_time,
                voxel_count=len(fused_data)
            )
            
            logger.info(f"Porosity analysis completed: {len(porosity_clusters)} clusters found in {analysis_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in porosity analysis: {e}")
            return PorosityAnalysisResult(
                success=False,
                porosity_statistics={},
                porosity_clusters=[],
                spatial_distribution={},
                process_correlations={},
                analysis_time=0.0,
                voxel_count=0,
                error_message=str(e)
            )
    
    def _extract_porosity_data(
        self, 
        fused_data: Dict[Tuple[int, int, int], FusedVoxelData], 
        dimensions: Tuple[int, int, int]
    ) -> Dict[str, np.ndarray]:
        """Extract porosity-related data from fused voxel data."""
        # Initialize arrays
        porosity_map = np.zeros(dimensions, dtype=np.float32)
        density_map = np.zeros(dimensions, dtype=np.float32)
        quality_map = np.zeros(dimensions, dtype=np.float32)
        temperature_map = np.zeros(dimensions, dtype=np.float32)
        laser_power_map = np.zeros(dimensions, dtype=np.float32)
        scan_speed_map = np.zeros(dimensions, dtype=np.float32)
        layer_map = np.zeros(dimensions, dtype=np.int16)
        voxel_mask = np.zeros(dimensions, dtype=bool)
        
        # Fill arrays with data
        for voxel_idx, voxel_data in fused_data.items():
            if self._is_valid_voxel_index(voxel_idx, dimensions):
                porosity_map[voxel_idx] = voxel_data.ct_porosity or 0.0
                density_map[voxel_idx] = voxel_data.ct_density or 4.43  # Ti-6Al-4V density
                quality_map[voxel_idx] = voxel_data.overall_quality_score or 100.0
                temperature_map[voxel_idx] = voxel_data.ispm_temperature or 0.0
                laser_power_map[voxel_idx] = voxel_data.laser_power
                scan_speed_map[voxel_idx] = voxel_data.scan_speed
                layer_map[voxel_idx] = voxel_data.layer_number
                voxel_mask[voxel_idx] = True
        
        return {
            'porosity': porosity_map,
            'density': density_map,
            'quality': quality_map,
            'temperature': temperature_map,
            'laser_power': laser_power_map,
            'scan_speed': scan_speed_map,
            'layer': layer_map,
            'voxel_mask': voxel_mask
        }
    
    def _calculate_porosity_statistics(self, porosity_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Calculate comprehensive porosity statistics."""
        porosity_map = porosity_data['porosity']
        voxel_mask = porosity_data['voxel_mask']
        
        # Get porosity values for solid voxels
        porosity_values = porosity_map[voxel_mask]
        
        if len(porosity_values) == 0:
            return {
                'mean_porosity': 0.0,
                'std_porosity': 0.0,
                'min_porosity': 0.0,
                'max_porosity': 0.0,
                'porosity_percentiles': {},
                'porosity_distribution': {},
                'total_porous_voxels': 0,
                'porosity_percentage': 0.0
            }
        
        # Basic statistics
        mean_porosity = np.mean(porosity_values)
        std_porosity = np.std(porosity_values)
        min_porosity = np.min(porosity_values)
        max_porosity = np.max(porosity_values)
        
        # Percentiles
        percentiles = {
            25: np.percentile(porosity_values, 25),
            50: np.percentile(porosity_values, 50),
            75: np.percentile(porosity_values, 75),
            90: np.percentile(porosity_values, 90),
            95: np.percentile(porosity_values, 95),
            99: np.percentile(porosity_values, 99)
        }
        
        # Porosity distribution
        porous_voxels = np.sum(porosity_values > self.config.porosity_threshold)
        high_porous_voxels = np.sum(porosity_values > self.config.high_porosity_threshold)
        critical_porous_voxels = np.sum(porosity_values > self.config.critical_porosity_threshold)
        
        porosity_distribution = {
            'low_porosity': np.sum((porosity_values > 0) & (porosity_values <= self.config.porosity_threshold)),
            'moderate_porosity': np.sum((porosity_values > self.config.porosity_threshold) & (porosity_values <= self.config.high_porosity_threshold)),
            'high_porosity': np.sum((porosity_values > self.config.high_porosity_threshold) & (porosity_values <= self.config.critical_porosity_threshold)),
            'critical_porosity': critical_porous_voxels
        }
        
        return {
            'mean_porosity': mean_porosity,
            'std_porosity': std_porosity,
            'min_porosity': min_porosity,
            'max_porosity': max_porosity,
            'porosity_percentiles': percentiles,
            'porosity_distribution': porosity_distribution,
            'total_porous_voxels': porous_voxels,
            'porosity_percentage': porous_voxels / len(porosity_values) * 100
        }
    
    def _analyze_porosity_clusters(
        self, 
        porosity_data: Dict[str, np.ndarray], 
        voxel_size: float
    ) -> List[PorosityCluster]:
        """Analyze porosity clusters."""
        porosity_map = porosity_data['porosity']
        voxel_mask = porosity_data['voxel_mask']
        
        # Find high porosity regions
        high_porosity_mask = (porosity_map > self.config.porosity_threshold) & voxel_mask
        
        if not np.any(high_porosity_mask):
            return []
        
        # Find connected components
        labeled_porosity, num_features = ndimage.label(
            high_porosity_mask, 
            structure=ndimage.generate_binary_structure(3, self.config.connectivity)
        )
        
        porosity_clusters = []
        
        for i in range(1, num_features + 1):
            porosity_region = (labeled_porosity == i)
            porosity_voxels = np.where(porosity_region)
            
            if len(porosity_voxels[0]) < self.config.min_porosity_cluster_size:
                continue
            
            if len(porosity_voxels[0]) > self.config.max_porosity_cluster_size:
                continue
            
            # Calculate cluster characteristics
            cluster_porosity_values = porosity_map[porosity_region]
            
            # Calculate centroid
            centroid = self._calculate_centroid(porosity_voxels, voxel_size)
            
            # Calculate volume
            volume = len(porosity_voxels[0]) * (voxel_size ** 3)
            
            # Calculate porosity statistics
            avg_porosity = np.mean(cluster_porosity_values)
            max_porosity = np.max(cluster_porosity_values)
            min_porosity = np.min(cluster_porosity_values)
            porosity_std = np.std(cluster_porosity_values)
            
            # Calculate geometric properties
            sphericity = self._calculate_sphericity(porosity_voxels, voxel_size)
            aspect_ratio = self._calculate_aspect_ratio(porosity_voxels, voxel_size)
            surface_area = self._calculate_surface_area(porosity_voxels, voxel_size)
            equivalent_diameter = self._calculate_equivalent_diameter(volume)
            
            # Create porosity cluster
            cluster = PorosityCluster(
                cluster_id=f"porosity_cluster_{i}",
                voxel_indices=list(zip(porosity_voxels[0], porosity_voxels[1], porosity_voxels[2])),
                centroid=centroid,
                volume=volume,
                porosity=avg_porosity,
                max_porosity=max_porosity,
                min_porosity=min_porosity,
                porosity_std=porosity_std,
                sphericity=sphericity,
                aspect_ratio=aspect_ratio,
                surface_area=surface_area,
                equivalent_diameter=equivalent_diameter
            )
            
            porosity_clusters.append(cluster)
        
        return porosity_clusters
    
    def _analyze_spatial_distribution(
        self, 
        porosity_data: Dict[str, np.ndarray], 
        voxel_size: float
    ) -> Dict[str, Any]:
        """Analyze spatial distribution of porosity."""
        porosity_map = porosity_data['porosity']
        voxel_mask = porosity_data['voxel_mask']
        layer_map = porosity_data['layer']
        
        # Layer-wise porosity analysis
        layer_porosity = {}
        unique_layers = np.unique(layer_map[voxel_mask])
        
        for layer in unique_layers:
            layer_mask = (layer_map == layer) & voxel_mask
            if np.any(layer_mask):
                layer_porosity_values = porosity_map[layer_mask]
                layer_porosity[layer] = {
                    'mean_porosity': np.mean(layer_porosity_values),
                    'std_porosity': np.std(layer_porosity_values),
                    'max_porosity': np.max(layer_porosity_values),
                    'porous_voxels': np.sum(layer_porosity_values > self.config.porosity_threshold),
                    'total_voxels': np.sum(layer_mask)
                }
        
        # Spatial autocorrelation
        spatial_autocorrelation = self._calculate_spatial_autocorrelation(porosity_map, voxel_mask)
        
        # Porosity gradient
        porosity_gradient = self._calculate_porosity_gradient(porosity_map, voxel_size)
        
        return {
            'layer_porosity': layer_porosity,
            'spatial_autocorrelation': spatial_autocorrelation,
            'porosity_gradient': porosity_gradient,
            'porosity_variance': np.var(porosity_map[voxel_mask])
        }
    
    def _analyze_process_correlations(
        self, 
        fused_data: Dict[Tuple[int, int, int], FusedVoxelData],
        porosity_data: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Analyze correlations between porosity and process parameters."""
        correlations = {}
        
        # Extract data for correlation analysis
        porosity_values = []
        laser_power_values = []
        scan_speed_values = []
        temperature_values = []
        quality_values = []
        
        for voxel_data in fused_data.values():
            if voxel_data.ct_porosity is not None:
                porosity_values.append(voxel_data.ct_porosity)
                laser_power_values.append(voxel_data.laser_power)
                scan_speed_values.append(voxel_data.scan_speed)
                temperature_values.append(voxel_data.ispm_temperature or 0.0)
                quality_values.append(voxel_data.overall_quality_score or 100.0)
        
        if len(porosity_values) < 10:  # Minimum sample size
            return correlations
        
        # Calculate correlations
        try:
            # Porosity vs Laser Power
            if len(set(laser_power_values)) > 1:
                corr, p_value = pearsonr(porosity_values, laser_power_values)
                if p_value < self.config.significance_level:
                    correlations['laser_power'] = corr
            
            # Porosity vs Scan Speed
            if len(set(scan_speed_values)) > 1:
                corr, p_value = pearsonr(porosity_values, scan_speed_values)
                if p_value < self.config.significance_level:
                    correlations['scan_speed'] = corr
            
            # Porosity vs Temperature
            if len(set(temperature_values)) > 1:
                corr, p_value = pearsonr(porosity_values, temperature_values)
                if p_value < self.config.significance_level:
                    correlations['temperature'] = corr
            
            # Porosity vs Quality
            if len(set(quality_values)) > 1:
                corr, p_value = pearsonr(porosity_values, quality_values)
                if p_value < self.config.significance_level:
                    correlations['quality'] = corr
        
        except Exception as e:
            logger.warning(f"Error calculating correlations: {e}")
        
        return correlations
    
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
    
    def _calculate_sphericity(self, voxel_coords: Tuple[np.ndarray, np.ndarray, np.ndarray], voxel_size: float) -> float:
        """Calculate sphericity of porosity cluster."""
        # Calculate volume
        volume = len(voxel_coords[0]) * (voxel_size ** 3)
        
        # Calculate surface area
        surface_area = self._calculate_surface_area(voxel_coords, voxel_size)
        
        # Sphericity = (π^(1/3) * (6V)^(2/3)) / A
        if surface_area > 0:
            sphericity = (np.pi ** (1/3) * (6 * volume) ** (2/3)) / surface_area
            return min(1.0, sphericity)  # Sphericity cannot exceed 1
        else:
            return 0.0
    
    def _calculate_aspect_ratio(self, voxel_coords: Tuple[np.ndarray, np.ndarray, np.ndarray], voxel_size: float) -> float:
        """Calculate aspect ratio of porosity cluster."""
        # Calculate bounding box dimensions
        x_range = (np.max(voxel_coords[0]) - np.min(voxel_coords[0])) * voxel_size
        y_range = (np.max(voxel_coords[1]) - np.min(voxel_coords[1])) * voxel_size
        z_range = (np.max(voxel_coords[2]) - np.min(voxel_coords[2])) * voxel_size
        
        # Aspect ratio = max dimension / min dimension
        dimensions = [x_range, y_range, z_range]
        return np.max(dimensions) / np.min(dimensions) if np.min(dimensions) > 0 else 1.0
    
    def _calculate_surface_area(self, voxel_coords: Tuple[np.ndarray, np.ndarray, np.ndarray], voxel_size: float) -> float:
        """Calculate surface area of porosity cluster."""
        # Simple approximation: number of voxels * voxel face area
        return len(voxel_coords[0]) * 6 * (voxel_size ** 2)
    
    def _calculate_equivalent_diameter(self, volume: float) -> float:
        """Calculate equivalent spherical diameter."""
        # D = (6V/π)^(1/3)
        return (6 * volume / np.pi) ** (1/3)
    
    def _calculate_spatial_autocorrelation(self, porosity_map: np.ndarray, voxel_mask: np.ndarray) -> float:
        """Calculate spatial autocorrelation of porosity."""
        # Simplified Moran's I calculation
        valid_indices = np.where(voxel_mask)
        
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
        
        values = porosity_map[valid_indices[0][sample_indices], 
                            valid_indices[1][sample_indices], 
                            valid_indices[2][sample_indices]]
        
        # Calculate distance matrix
        from scipy.spatial import distance_matrix
        distances = distance_matrix(positions, positions)
        
        # Calculate autocorrelation (simplified)
        mean_value = np.mean(values)
        numerator = 0
        denominator = 0
        
        for i in range(len(values)):
            for j in range(len(values)):
                if i != j and distances[i, j] <= 2:  # Neighboring voxels
                    numerator += (values[i] - mean_value) * (values[j] - mean_value)
            denominator += (values[i] - mean_value) ** 2
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _calculate_porosity_gradient(self, porosity_map: np.ndarray, voxel_size: float) -> np.ndarray:
        """Calculate porosity gradient."""
        # Apply Gaussian smoothing
        smoothed_porosity = ndimage.gaussian_filter(porosity_map, sigma=1.0)
        
        # Calculate gradient
        gradient = np.gradient(smoothed_porosity, voxel_size)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(
            gradient[0]**2 + gradient[1]**2 + gradient[2]**2
        )
        
        return gradient_magnitude
    
    def generate_porosity_report(self, result: PorosityAnalysisResult) -> str:
        """Generate comprehensive porosity analysis report."""
        report = []
        report.append("=== POROSITY ANALYSIS REPORT ===\n")
        
        # Overall statistics
        stats = result.porosity_statistics
        report.append("OVERALL POROSITY STATISTICS:")
        report.append(f"  Mean Porosity: {stats['mean_porosity']:.4f}")
        report.append(f"  Standard Deviation: {stats['std_porosity']:.4f}")
        report.append(f"  Min Porosity: {stats['min_porosity']:.4f}")
        report.append(f"  Max Porosity: {stats['max_porosity']:.4f}")
        report.append(f"  Porosity Percentage: {stats['porosity_percentage']:.2f}%")
        report.append("")
        
        # Porosity distribution
        report.append("POROSITY DISTRIBUTION:")
        for category, count in stats['porosity_distribution'].items():
            report.append(f"  {category.replace('_', ' ').title()}: {count} voxels")
        report.append("")
        
        # Porosity clusters
        report.append("POROSITY CLUSTERS:")
        report.append(f"  Total Clusters: {len(result.porosity_clusters)}")
        for i, cluster in enumerate(result.porosity_clusters[:10]):  # Show first 10
            report.append(f"  Cluster {i+1}:")
            report.append(f"    Volume: {cluster.volume:.4f} mm³")
            report.append(f"    Porosity: {cluster.porosity:.4f}")
            report.append(f"    Sphericity: {cluster.sphericity:.4f}")
            report.append(f"    Aspect Ratio: {cluster.aspect_ratio:.4f}")
        report.append("")
        
        # Process correlations
        report.append("PROCESS CORRELATIONS:")
        for param, correlation in result.process_correlations.items():
            report.append(f"  {param.replace('_', ' ').title()}: {correlation:.4f}")
        report.append("")
        
        # Spatial distribution
        spatial = result.spatial_distribution
        report.append("SPATIAL DISTRIBUTION:")
        report.append(f"  Spatial Autocorrelation: {spatial['spatial_autocorrelation']:.4f}")
        report.append(f"  Porosity Variance: {spatial['porosity_variance']:.4f}")
        report.append(f"  Layers Analyzed: {len(spatial['layer_porosity'])}")
        report.append("")
        
        report.append("=== END OF REPORT ===")
        
        return "\n".join(report)
    
    def export_analysis_results(self, result: PorosityAnalysisResult, output_path: str):
        """Export porosity analysis results to file."""
        try:
            import json
            from pathlib import Path
            
            # Create output directory if it doesn't exist
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare data for export
            export_data = {
                'porosity_statistics': result.porosity_statistics,
                'porosity_clusters': [
                    {
                        'cluster_id': cluster.cluster_id,
                        'centroid': cluster.centroid,
                        'volume': cluster.volume,
                        'porosity': cluster.porosity,
                        'max_porosity': cluster.max_porosity,
                        'min_porosity': cluster.min_porosity,
                        'porosity_std': cluster.porosity_std,
                        'sphericity': cluster.sphericity,
                        'aspect_ratio': cluster.aspect_ratio,
                        'surface_area': cluster.surface_area,
                        'equivalent_diameter': cluster.equivalent_diameter,
                        'voxel_count': len(cluster.voxel_indices)
                    }
                    for cluster in result.porosity_clusters
                ],
                'spatial_distribution': result.spatial_distribution,
                'process_correlations': result.process_correlations,
                'analysis_metadata': {
                    'analysis_time': result.analysis_time,
                    'voxel_count': result.voxel_count,
                    'cluster_count': len(result.porosity_clusters),
                    'analysis_timestamp': datetime.now().isoformat()
                }
            }
            
            # Export to JSON
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Porosity analysis results exported to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting porosity analysis results: {e}")
            raise
