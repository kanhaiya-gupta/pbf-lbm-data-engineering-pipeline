"""
Voxel Visualization Clients for PBF-LB/M Data Pipeline

This module provides comprehensive voxel-level visualization and analysis capabilities
for PBF-LB/M (Powder Bed Fusion - Laser Beam/Metal) additive manufacturing research.
It enables spatially-resolved process control and quality analysis at the voxel level.

Key Features:
- CAD model voxelization
- Multi-modal data fusion (ISPM, CT, process parameters)
- Voxel-level process control
- Spatially-resolved quality analysis
- Real-time visualization
- Defect detection and analysis
- Process optimization
- Interactive 3D visualization
- Data export and sharing
- Web-based dashboard

Architecture:
- Core: Voxelization, fusion, control, rendering, and loading systems
- Analysis: Quality analysis, defect detection, and porosity analysis
- Interaction: User interface and controls
- Export: Data export and sharing
- Web: Web-based visualization dashboard
"""

# Core components
from .core.cad_voxelizer import CADVoxelizer, VoxelGrid, VoxelizationConfig
from .core.multi_modal_fusion import MultiModalFusion, FusedVoxelData
from .core.voxel_process_controller import VoxelProcessController, ProcessControlConfig, ControlMode, OptimizationObjective
from .core.voxel_renderer import VoxelRenderer, RenderConfig, RenderResult
from .core.voxel_loader import VoxelLoader, LoadConfig, LoadResult

# Analysis components
from .analysis.spatial_quality_analyzer import SpatialQualityAnalyzer, SpatialQualityMetrics, QualityAnalysisConfig
from .analysis.defect_detector_3d import DefectDetector3D, DefectDetectionConfig, DefectDetectionResult, DefectType
from .analysis.porosity_analyzer import PorosityAnalyzer, PorosityAnalysisConfig, PorosityAnalysisResult, PorosityCluster

# Interaction components
from .interaction.voxel_controller import VoxelController, InteractionConfig, InteractionMode, SelectionType, VoxelSelection, InteractionEvent

# Export components
from .export.voxel_exporter import VoxelExporter, ExportConfig, ExportResult

__all__ = [
    # Core components
    'CADVoxelizer',
    'VoxelGrid', 
    'VoxelizationConfig',
    'MultiModalFusion',
    'FusedVoxelData',
    'VoxelProcessController',
    'ProcessControlConfig',
    'ControlMode',
    'OptimizationObjective',
    'VoxelRenderer',
    'RenderConfig',
    'RenderResult',
    'VoxelLoader',
    'LoadConfig',
    'LoadResult',
    
    # Analysis components
    'SpatialQualityAnalyzer',
    'SpatialQualityMetrics',
    'QualityAnalysisConfig',
    'DefectDetector3D',
    'DefectDetectionConfig',
    'DefectDetectionResult',
    'DefectType',
    'PorosityAnalyzer',
    'PorosityAnalysisConfig',
    'PorosityAnalysisResult',
    'PorosityCluster',
    
    # Interaction components
    'VoxelController',
    'InteractionConfig',
    'InteractionMode',
    'SelectionType',
    'VoxelSelection',
    'InteractionEvent',
    
    # Export components
    'VoxelExporter',
    'ExportConfig',
    'ExportResult',
]

# Version information
__version__ = "1.0.0"
__author__ = "PBF-LB/M Research Team"
__description__ = "Voxel-level visualization and analysis for PBF-LB/M additive manufacturing"
