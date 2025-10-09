"""
Multi-Modal Data Fusion for PBF-LB/M Voxel Analysis

This module provides comprehensive multi-modal data fusion capabilities that integrate
CAD voxel models with ISPM (In-Situ Process Monitoring) data, CT scan data, and
process parameters for spatially-resolved analysis in PBF-LB/M research.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from datetime import datetime
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import cv2

from src.core.domain.value_objects.voxel_coordinates import VoxelCoordinates
from src.core.domain.entities.ispm_monitoring import ISPMMonitoring
from src.core.domain.entities.ct_scan import CTScan
from src.core.domain.value_objects.quality_metrics import QualityMetrics
from src.core.domain.value_objects.defect_classification import DefectClassification

from .cad_voxelizer import VoxelGrid

logger = logging.getLogger(__name__)


@dataclass
class ISPMDataPoint:
    """ISPM data point with spatial and temporal information."""
    
    timestamp: datetime
    position: Tuple[float, float, float]  # (x, y, z) in mm
    sensor_type: str
    measurement_value: float
    unit: str
    confidence: float
    quality_score: float


@dataclass
class CTDataPoint:
    """CT scan data point with spatial information."""
    
    position: Tuple[float, float, float]  # (x, y, z) in mm
    density: float  # Hounsfield units or relative density
    intensity: float  # CT intensity value
    defect_probability: float
    material_type: str
    quality_score: float


@dataclass
class FusedVoxelData:
    """Fused voxel data combining CAD, ISPM, and CT information."""
    
    # CAD geometry
    voxel_coordinates: VoxelCoordinates
    is_solid: bool
    material_type: str
    
    # Process parameters
    laser_power: float
    scan_speed: float
    layer_number: int
    build_time: datetime
    
    # Energy parameters (for cost analysis)
    energy_density: Optional[float] = None
    exposure_time: Optional[float] = None
    total_energy_input: Optional[float] = None
    
    # ISPM data
    ispm_temperature: Optional[float] = None
    ispm_melt_pool_size: Optional[float] = None
    ispm_acoustic_emissions: Optional[float] = None
    ispm_plume_intensity: Optional[float] = None
    ispm_confidence: Optional[float] = None
    
    # CT scan data
    ct_density: Optional[float] = None
    ct_intensity: Optional[float] = None
    ct_defect_probability: Optional[float] = None
    ct_porosity: Optional[float] = None
    
    # Quality metrics
    overall_quality_score: float = 100.0
    dimensional_accuracy: Optional[float] = None
    surface_roughness: Optional[float] = None
    defect_count: int = 0
    defect_types: List[str] = None
    
    # Fusion metadata
    fusion_confidence: float = 1.0
    data_completeness: float = 1.0
    last_updated: datetime = None


class MultiModalFusion:
    """
    Multi-modal data fusion system for PBF-LB/M voxel analysis.
    
    This class provides comprehensive data fusion capabilities including:
    - ISPM data registration to voxel coordinates
    - CT scan data alignment with voxel grid
    - Quality metrics integration
    - Defect detection and classification
    - Spatially-resolved analysis
    """
    
    def __init__(self, fusion_config: Dict = None):
        """Initialize the multi-modal fusion system."""
        self.config = fusion_config or self._default_config()
        self.spatial_tolerance = self.config.get('spatial_tolerance', 0.1)  # mm
        self.temporal_tolerance = self.config.get('temporal_tolerance', 1.0)  # seconds
        self.fusion_weights = self.config.get('fusion_weights', {
            'ispm': 0.4,
            'ct': 0.4,
            'process': 0.2
        })
        
        logger.info("Multi-Modal Fusion system initialized")
    
    def _default_config(self) -> Dict:
        """Default fusion configuration."""
        return {
            'spatial_tolerance': 0.1,  # mm
            'temporal_tolerance': 1.0,  # seconds
            'fusion_weights': {
                'ispm': 0.4,
                'ct': 0.4,
                'process': 0.2
            },
            'interpolation_method': 'linear',
            'quality_threshold': 0.8,
            'defect_detection_enabled': True
        }
    
    def register_build_data(
        self,
        voxel_grid: VoxelGrid,
        build_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Register build file data (SLM, CLI, etc.) with voxel grid.
        
        This method processes build file data and maps process parameters
        to voxel coordinates for later fusion operations.
        
        Args:
            voxel_grid: Voxelized CAD model
            build_data: Dictionary containing build file data (SLM, CLI, etc.)
            
        Returns:
            Dictionary containing registered build data mapped to voxel coordinates
        """
        logger.info("Registering build data with voxel grid...")
        
        registered_data = {
            'voxel_grid': voxel_grid,
            'build_files': {},
            'process_parameters': {},
            'layer_data': {},
            'extracted_data': {},  # NEW: Store all extracted data from build parsers
            'registration_metadata': {
                'timestamp': datetime.now(),
                'voxel_count': voxel_grid.total_voxels,
                'solid_voxels': voxel_grid.solid_voxels
            }
        }
        
        # Process each build file type
        for file_type, file_data in build_data.items():
            if file_data is None:
                continue
                
            logger.info(f"Processing {file_type.upper()} build data...")
            
            # Extract process parameters from build data
            if 'layers' in file_data:
                registered_data['layer_data'][file_type] = file_data['layers']
            
            if 'parameters' in file_data:
                registered_data['process_parameters'][file_type] = file_data['parameters']
            
            # Store the REAL extracted data from our build parsers
            if 'extracted_data' in file_data:
                registered_data['extracted_data'][file_type] = file_data['extracted_data']
                logger.info(f"Stored extracted data for {file_type.upper()}: {list(file_data['extracted_data'].keys())}")
            
            # Store the full build data
            registered_data['build_files'][file_type] = file_data
            
            logger.info(f"Successfully registered {file_type.upper()} data")
        
        logger.info(f"Build data registration completed: {len(registered_data['build_files'])} files processed")
        return registered_data
    
    def fuse_voxel_data(
        self,
        voxel_grid: VoxelGrid,
        ispm_data: List[ISPMMonitoring],
        ct_data: List[CTScan],
        quality_metrics: Optional[List[QualityMetrics]] = None,
        registered_build_data: Optional[Dict[str, Any]] = None,
        layer_thickness: float = 0.05
    ) -> Dict[Tuple[int, int, int], FusedVoxelData]:
        """
        Fuse CAD voxel data with ISPM, CT scan data, and build file data.
        
        Args:
            voxel_grid: Voxelized CAD model
            ispm_data: ISPM monitoring data
            ct_data: CT scan data
            quality_metrics: Quality metrics data
            registered_build_data: Previously registered build file data
            layer_thickness: Layer thickness in mm (from application layer)
            
        Returns:
            Dict mapping voxel indices to fused voxel data
        """
        try:
            logger.info("Starting multi-modal data fusion...")
            
            # Initialize fused data dictionary
            fused_data = {}
            
            # Get all solid voxels - handle both VoxelGrid object and dict
            if hasattr(voxel_grid, 'voxels'):
                voxels_array = voxel_grid.voxels
            elif isinstance(voxel_grid, dict) and 'voxels' in voxel_grid:
                voxels_array = voxel_grid.voxels
            else:
                raise ValueError("Invalid voxel_grid format")
            
            solid_voxels = np.where(voxels_array > 0)
            
            # Process each solid voxel
            for i in range(len(solid_voxels[0])):
                voxel_idx = (solid_voxels[0][i], solid_voxels[1][i], solid_voxels[2][i])
                
                # Create base voxel data from CAD model
                base_data = self._create_base_voxel_data(voxel_grid, voxel_idx)
                
                # Fuse build data if available
                if registered_build_data:
                    base_data = self._fuse_build_data(base_data, registered_build_data, voxel_idx, layer_thickness)
                
                # Fuse ISPM data
                ispm_fused = self._fuse_ispm_data(base_data, ispm_data)
                
                # Fuse CT scan data
                ct_fused = self._fuse_ct_data(ispm_fused, ct_data)
                
                # Fuse quality metrics
                if quality_metrics:
                    ct_fused = self._fuse_quality_metrics(ct_fused, quality_metrics)
                
                # Calculate fusion confidence and completeness
                ct_fused = self._calculate_fusion_metrics(ct_fused)
                
                # Store fused data
                fused_data[voxel_idx] = ct_fused
            
            logger.info(f"Multi-modal fusion completed: {len(fused_data)} voxels processed")
            return fused_data
            
        except Exception as e:
            logger.error(f"Error in multi-modal data fusion: {e}")
            raise
    
    def _fuse_build_data(
        self,
        base_data: FusedVoxelData,
        registered_build_data: Dict[str, Any],
        voxel_idx: Tuple[int, int, int],
        layer_thickness: float
    ) -> FusedVoxelData:
        """
        Fuse build file data (SLM, CLI, etc.) with voxel data using REAL extracted data.
        
        Args:
            base_data: Base voxel data from CAD model
            registered_build_data: Registered build file data with extracted parameters
            voxel_idx: Voxel coordinates
            layer_thickness: Layer thickness in mm (from application layer)
            
        Returns:
            Enhanced voxel data with REAL build file information
        """
        try:
            # Convert voxel index to world coordinates
            voxel_grid = registered_build_data.get('voxel_grid', {})
            bounds = voxel_grid.bounds
            voxel_size = voxel_grid.voxel_size
            
            if len(bounds) >= 3:
                world_x = bounds[0][0] + voxel_idx[0] * voxel_size
                world_y = bounds[1][0] + voxel_idx[1] * voxel_size
                world_z = bounds[2][0] + voxel_idx[2] * voxel_size
                
                # Get REAL extracted data from our build parsers
                extracted_data = registered_build_data.get('extracted_data', {})
                process_params = registered_build_data.get('process_parameters', {})
                layer_data = registered_build_data.get('layer_data', {})
                
                # Map voxel coordinates to process parameters using spatial lookup
                voxel_process_data = self._map_voxel_to_process_parameters(
                    (world_x, world_y, world_z), 
                    extracted_data, 
                    voxel_size,
                    layer_thickness
                )
                
                # Update base_data with REAL extracted parameters
                if voxel_process_data:
                    # Power data from power_extractor
                    if 'power_data' in voxel_process_data:
                        power_info = voxel_process_data['power_data']
                        base_data.laser_power = power_info.get('power', 0.0)
                    
                    # Velocity data from velocity_extractor
                    if 'velocity_data' in voxel_process_data:
                        velocity_info = voxel_process_data['velocity_data']
                        base_data.scan_speed = velocity_info.get('velocity', 0.0)
                    
                    # Layer data from layer_extractor
                    if 'layer_data' in voxel_process_data:
                        layer_info = voxel_process_data['layer_data']
                        base_data.layer_number = layer_info.get('layer_number', 0)
                    
                    # Timestamp data from timestamp_extractor
                    if 'timestamp_data' in voxel_process_data:
                        timestamp_info = voxel_process_data['timestamp_data']
                        base_data.build_time = timestamp_info.get('build_time', datetime.now())
                    
                    # Energy data from energy_extractor (for cost analysis)
                    if 'energy_data' in voxel_process_data:
                        energy_info = voxel_process_data['energy_data']
                        base_data.energy_density = energy_info.get('energy_density', None)
                        base_data.exposure_time = energy_info.get('exposure_time', None)
                        base_data.total_energy_input = energy_info.get('total_energy_input', None)
                    
                    # Update material type if available
                    if 'material_type' in voxel_process_data:
                        base_data.material_type = voxel_process_data['material_type']
                    
                    # Store additional process parameters in build_file_data for granularity processing
                    base_data.build_file_data = {
                        'world_coordinates': (world_x, world_y, world_z),
                        'voxel_coordinates': voxel_idx,
                        'process_parameters': process_params,
                        'layer_data': layer_data,
                        'build_files': registered_build_data.get('build_files', {}),
                        'extracted_data': extracted_data,
                        'voxel_process_mapping': voxel_process_data
                    }
                
                logger.debug(f"Fused REAL build data for voxel {voxel_idx}: {voxel_process_data}")
            
            return base_data
            
        except Exception as e:
            logger.warning(f"Error fusing build data for voxel {voxel_idx}: {e}")
            return base_data
    
    def _map_voxel_to_process_parameters(
        self, 
        world_coords: Tuple[float, float, float], 
        extracted_data: Dict[str, Any], 
        voxel_size: float,
        layer_thickness: float = 0.05
    ) -> Dict[str, Any]:
        """
        Map voxel world coordinates to process parameters using REAL data structures.
        
        Args:
            world_coords: (x, y, z) world coordinates of the voxel
            extracted_data: All extracted data from build parsers
            voxel_size: Size of each voxel for spatial matching
            layer_thickness: Layer thickness in mm (passed from application layer)
            
        Returns:
            Process parameters mapped to this voxel
        """
        try:
            voxel_x, voxel_y, voxel_z = world_coords
            voxel_process_data = {}
            
            # Layer thickness should be passed as parameter from application layer
            # This framework should not hardcode or assume specific values
            
            # Spatial tolerance for matching coordinates
            tolerance = voxel_size * 0.5
            
            # Map power data using REAL data structure
            if 'power_data' in extracted_data:
                power_data = extracted_data['power_data']
                
                # Use hatch_power data (real structure from power_extractor)
                if 'hatch_power' in power_data:
                    for hatch in power_data['hatch_power']:
                        if hatch.get('coordinates') and hatch.get('power'):
                            coords = hatch['coordinates']
                            # coords is a list of [x1, y1, x2, y2, ...] pairs
                            for i in range(0, len(coords), 2):
                                if i + 1 < len(coords):
                                    hatch_x, hatch_y = coords[i], coords[i + 1]
                                    hatch_z = hatch.get('layer_index', 0) * layer_thickness  # Convert layer to Z using real layer thickness
                                    
                                    if self._is_coordinate_match((voxel_x, voxel_y, voxel_z), 
                                                               (hatch_x, hatch_y, hatch_z), tolerance):
                                        voxel_process_data['power_data'] = {
                                            'power': hatch.get('power'),
                                            'velocity': hatch.get('velocity'),
                                            'exposure_time': hatch.get('exposure_time'),
                                            'point_distance': hatch.get('point_distance'),
                                            'laser_focus': hatch.get('laser_focus'),
                                            'coordinates': (hatch_x, hatch_y, hatch_z)
                                        }
                                        break
                
                # Use spatial_power_map data (real structure)
                if 'spatial_power_map' in power_data and not voxel_process_data.get('power_data'):
                    spatial_map = power_data['spatial_power_map']
                    if 'power_voxels' in spatial_map:
                        for power_point in spatial_map['power_voxels']:
                            if self._is_coordinate_match((voxel_x, voxel_y, voxel_z),
                                                       (power_point['x'], power_point['y'], power_point['z']), 
                                                       tolerance):
                                voxel_process_data['power_data'] = {
                                    'power': power_point.get('power'),
                                    'velocity': power_point.get('velocity'),
                                    'geometry_type': power_point.get('geometry_type'),
                                    'coordinates': (power_point['x'], power_point['y'], power_point['z'])
                                }
                                break
            
            # Map velocity data using REAL data structure
            if 'velocity_data' in extracted_data:
                velocity_data = extracted_data['velocity_data']
                
                # Use hatch_velocity data (real structure from velocity_extractor)
                if 'hatch_velocity' in velocity_data:
                    for hatch in velocity_data['hatch_velocity']:
                        if hatch.get('coordinates') and hatch.get('velocity'):
                            coords = hatch['coordinates']
                            for i in range(0, len(coords), 2):
                                if i + 1 < len(coords):
                                    hatch_x, hatch_y = coords[i], coords[i + 1]
                                    hatch_z = hatch.get('layer_index', 0) * layer_thickness  # Convert layer to Z using real layer thickness
                                    
                                    if self._is_coordinate_match((voxel_x, voxel_y, voxel_z),
                                                               (hatch_x, hatch_y, hatch_z), tolerance):
                                        voxel_process_data['velocity_data'] = {
                                            'velocity': hatch.get('velocity'),
                                            'power': hatch.get('power'),
                                            'exposure_time': hatch.get('exposure_time'),
                                            'coordinates': (hatch_x, hatch_y, hatch_z)
                                        }
                                        break
            
            # Map energy data using REAL data structure
            if 'energy_data' in extracted_data:
                energy_data = extracted_data['energy_data']
                
                # Use energy_density data (real structure from energy_extractor)
                if 'energy_density' in energy_data:
                    for energy_point in energy_data['energy_density']:
                        if energy_point.get('coordinates') and energy_point.get('energy_density'):
                            coords = energy_point['coordinates']
                            for i in range(0, len(coords), 2):
                                if i + 1 < len(coords):
                                    energy_x, energy_y = coords[i], coords[i + 1]
                                    energy_z = energy_point.get('layer_index', 0) * layer_thickness  # Convert layer to Z using real layer thickness
                                    
                                    if self._is_coordinate_match((voxel_x, voxel_y, voxel_z),
                                                               (energy_x, energy_y, energy_z), tolerance):
                                        voxel_process_data['energy_data'] = {
                                            'energy_density': energy_point.get('energy_density'),
                                            'power': energy_point.get('power'),
                                            'velocity': energy_point.get('velocity'),
                                            'exposure_time': energy_point.get('exposure_time'),
                                            'coordinates': (energy_x, energy_y, energy_z)
                                        }
                                        break
            
            # Map other data using similar real structure approach
            # (path_data, layer_data, etc. would follow similar pattern)
            
            return voxel_process_data
            
        except Exception as e:
            logger.warning(f"Error mapping voxel to process parameters: {e}")
            return {}
    
    def _is_coordinate_match(
        self, 
        voxel_coords: Tuple[float, float, float], 
        process_coords: Tuple[float, float, float], 
        tolerance: float
    ) -> bool:
        """Check if voxel coordinates match process coordinates within tolerance."""
        if not process_coords or len(process_coords) < 3:
            return False
        
        dx = abs(voxel_coords[0] - process_coords[0])
        dy = abs(voxel_coords[1] - process_coords[1])
        dz = abs(voxel_coords[2] - process_coords[2])
        
        return dx <= tolerance and dy <= tolerance and dz <= tolerance
    
    def _create_base_voxel_data(
        self, 
        voxel_grid: VoxelGrid, 
        voxel_idx: Tuple[int, int, int]
    ) -> FusedVoxelData:
        """Create base voxel data from CAD model."""
        # Handle both VoxelGrid object and dict
        if hasattr(voxel_grid, 'origin'):
            origin = voxel_grid.origin
            voxel_size = voxel_grid.voxel_size
        elif isinstance(voxel_grid, dict):
            bounds = voxel_grid.bounds
            voxel_size = voxel_grid.voxel_size
            if len(bounds) >= 3:
                origin = (bounds[0][0], bounds[1][0], bounds[2][0])
            else:
                origin = (0.0, 0.0, 0.0)
        else:
            origin = (0.0, 0.0, 0.0)
            voxel_size = 1.0
        
        # Convert grid index to world coordinates
        world_coords = (
            origin[0] + voxel_idx[0] * voxel_size,
            origin[1] + voxel_idx[1] * voxel_size,
            origin[2] + voxel_idx[2] * voxel_size
        )
        
        # Create VoxelCoordinates object
        voxel_coords = VoxelCoordinates(
            x=world_coords[0],
            y=world_coords[1],
            z=world_coords[2],
            voxel_size=voxel_size,
            is_solid=True,
            layer_number=0,  # Default layer number
            material_type="Ti-6Al-4V",  # Default material
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Create base fused data with empty/default values
        # Real values will be populated by _fuse_build_data() method using extracted data
        base_data = FusedVoxelData(
            voxel_coordinates=voxel_coords,
            is_solid=True,
            material_type="Unknown",  # Will be determined from build data
            laser_power=None,  # Will be filled with real data from build parsers
            scan_speed=None,   # Will be filled with real data from build parsers
            layer_number=None, # Will be filled with real layer data
            build_time=None,   # Will be filled with real build time
            # Energy parameters for cost analysis
            energy_density=None,     # Will be filled with real energy density
            exposure_time=None,      # Will be filled with real exposure time
            total_energy_input=None, # Will be filled with real energy input
            last_updated=datetime.now()
        )
        
        return base_data
    
    def _fuse_ispm_data(
        self, 
        base_data: FusedVoxelData, 
        ispm_data: List[ISPMMonitoring]
    ) -> FusedVoxelData:
        """Fuse ISPM data with voxel data."""
        # Find ISPM data points near this voxel
        nearby_ispm = self._find_nearby_ispm_data(base_data.voxel_coordinates, ispm_data)
        
        if not nearby_ispm:
            return base_data
        
        # Aggregate ISPM measurements
        ispm_aggregated = self._aggregate_ispm_measurements(nearby_ispm)
        
        # Update base data with ISPM information
        base_data.ispm_temperature = ispm_aggregated.get('temperature')
        base_data.ispm_melt_pool_size = ispm_aggregated.get('melt_pool_size')
        base_data.ispm_acoustic_emissions = ispm_aggregated.get('acoustic_emissions')
        base_data.ispm_plume_intensity = ispm_aggregated.get('plume_intensity')
        base_data.ispm_confidence = ispm_aggregated.get('confidence')
        
        return base_data
    
    def _fuse_ct_data(
        self, 
        base_data: FusedVoxelData, 
        ct_data: List[CTScan]
    ) -> FusedVoxelData:
        """Fuse CT scan data with voxel data."""
        # Find CT data points near this voxel
        nearby_ct = self._find_nearby_ct_data(base_data.voxel_coordinates, ct_data)
        
        if not nearby_ct:
            return base_data
        
        # Aggregate CT measurements
        ct_aggregated = self._aggregate_ct_measurements(nearby_ct)
        
        # Update base data with CT information
        base_data.ct_density = ct_aggregated.get('density')
        base_data.ct_intensity = ct_aggregated.get('intensity')
        base_data.ct_defect_probability = ct_aggregated.get('defect_probability')
        base_data.ct_porosity = ct_aggregated.get('porosity')
        
        return base_data
    
    def _fuse_quality_metrics(
        self, 
        base_data: FusedVoxelData, 
        quality_metrics: List[QualityMetrics]
    ) -> FusedVoxelData:
        """Fuse quality metrics with voxel data."""
        # Find quality metrics relevant to this voxel
        relevant_metrics = self._find_relevant_quality_metrics(base_data, quality_metrics)
        
        if not relevant_metrics:
            return base_data
        
        # Aggregate quality metrics
        quality_aggregated = self._aggregate_quality_metrics(relevant_metrics)
        
        # Update base data with quality information
        base_data.overall_quality_score = quality_aggregated.get('overall_quality', 100.0)
        base_data.dimensional_accuracy = quality_aggregated.get('dimensional_accuracy')
        base_data.surface_roughness = quality_aggregated.get('surface_roughness')
        base_data.defect_count = quality_aggregated.get('defect_count', 0)
        base_data.defect_types = quality_aggregated.get('defect_types', [])
        
        return base_data
    
    def _find_nearby_ispm_data(
        self, 
        voxel_coords: VoxelCoordinates, 
        ispm_data: List[ISPMMonitoring]
    ) -> List[ISPMMonitoring]:
        """Find ISPM data points near the voxel coordinates."""
        nearby_data = []
        voxel_position = (voxel_coords.x, voxel_coords.y, voxel_coords.z)
        
        for ispm_point in ispm_data:
            # Calculate distance between voxel and ISPM point
            if hasattr(ispm_point, 'position'):
                distance = np.linalg.norm(
                    np.array(voxel_position) - np.array(ispm_point.position)
                )
                
                if distance <= self.spatial_tolerance:
                    nearby_data.append(ispm_point)
        
        return nearby_data
    
    def _find_nearby_ct_data(
        self, 
        voxel_coords: VoxelCoordinates, 
        ct_data: List[CTScan]
    ) -> List[CTScan]:
        """Find CT scan data points near the voxel coordinates."""
        nearby_data = []
        voxel_position = (voxel_coords.x, voxel_coords.y, voxel_coords.z)
        
        for ct_point in ct_data:
            # Calculate distance between voxel and CT point
            if hasattr(ct_point, 'position'):
                distance = np.linalg.norm(
                    np.array(voxel_position) - np.array(ct_point.position)
                )
                
                if distance <= self.spatial_tolerance:
                    nearby_data.append(ct_point)
        
        return nearby_data
    
    def _find_relevant_quality_metrics(
        self, 
        base_data: FusedVoxelData, 
        quality_metrics: List[QualityMetrics]
    ) -> List[QualityMetrics]:
        """Find quality metrics relevant to the voxel."""
        relevant_metrics = []
        
        for metric in quality_metrics:
            # Check if quality metric is relevant to this voxel
            # This could be based on spatial proximity, layer number, or other criteria
            if self._is_metric_relevant(base_data, metric):
                relevant_metrics.append(metric)
        
        return relevant_metrics
    
    def _is_metric_relevant(self, base_data: FusedVoxelData, metric: QualityMetrics) -> bool:
        """Check if a quality metric is relevant to the voxel."""
        # Simple relevance check - can be enhanced based on specific requirements
        return True  # Placeholder implementation
    
    def _aggregate_ispm_measurements(self, ispm_data: List[ISPMMonitoring]) -> Dict:
        """Aggregate ISPM measurements for a voxel."""
        if not ispm_data:
            return {}
        
        # Group measurements by sensor type
        sensor_measurements = {}
        for ispm_point in ispm_data:
            sensor_type = getattr(ispm_point, 'sensor_type', 'unknown')
            if sensor_type not in sensor_measurements:
                sensor_measurements[sensor_type] = []
            
            measurement_value = getattr(ispm_point, 'measurement_value', 0.0)
            confidence = getattr(ispm_point, 'confidence', 1.0)
            
            sensor_measurements[sensor_type].append({
                'value': measurement_value,
                'confidence': confidence
            })
        
        # Calculate weighted averages
        aggregated = {}
        for sensor_type, measurements in sensor_measurements.items():
            if measurements:
                # Weighted average based on confidence
                total_weight = sum(m['confidence'] for m in measurements)
                if total_weight > 0:
                    weighted_avg = sum(
                        m['value'] * m['confidence'] for m in measurements
                    ) / total_weight
                    aggregated[sensor_type] = weighted_avg
                    
                    # Average confidence
                    avg_confidence = sum(m['confidence'] for m in measurements) / len(measurements)
                    aggregated[f'{sensor_type}_confidence'] = avg_confidence
        
        return aggregated
    
    def _aggregate_ct_measurements(self, ct_data: List[CTScan]) -> Dict:
        """Aggregate CT scan measurements for a voxel."""
        if not ct_data:
            return {}
        
        # Extract measurements
        densities = []
        intensities = []
        defect_probabilities = []
        quality_scores = []
        
        for ct_point in ct_data:
            if hasattr(ct_point, 'density'):
                densities.append(ct_point.density)
            if hasattr(ct_point, 'intensity'):
                intensities.append(ct_point.intensity)
            if hasattr(ct_point, 'defect_probability'):
                defect_probabilities.append(ct_point.defect_probability)
            if hasattr(ct_point, 'quality_score'):
                quality_scores.append(ct_point.quality_score)
        
        # Calculate statistics
        aggregated = {}
        if densities:
            aggregated['density'] = np.mean(densities)
        if intensities:
            aggregated['intensity'] = np.mean(intensities)
        if defect_probabilities:
            aggregated['defect_probability'] = np.mean(defect_probabilities)
        if quality_scores:
            aggregated['quality_score'] = np.mean(quality_scores)
        
        # Calculate porosity from density
        if 'density' in aggregated:
            # Assuming theoretical density of Ti-6Al-4V is 4.43 g/cm³
            theoretical_density = 4.43
            aggregated['porosity'] = max(0, (theoretical_density - aggregated['density']) / theoretical_density)
        
        return aggregated
    
    def _aggregate_quality_metrics(self, quality_metrics: List[QualityMetrics]) -> Dict:
        """Aggregate quality metrics for a voxel."""
        if not quality_metrics:
            return {}
        
        # Extract quality measurements
        overall_qualities = []
        dimensional_accuracies = []
        surface_roughnesses = []
        defect_counts = []
        defect_types = []
        
        for metric in quality_metrics:
            if hasattr(metric, 'quality_score'):
                overall_qualities.append(metric.quality_score)
            if hasattr(metric, 'dimensional_accuracy'):
                dimensional_accuracies.append(metric.dimensional_accuracy)
            if hasattr(metric, 'surface_roughness_ra'):
                surface_roughnesses.append(metric.surface_roughness_ra)
            if hasattr(metric, 'defect_count'):
                defect_counts.append(metric.defect_count)
            if hasattr(metric, 'defect_types'):
                defect_types.extend(metric.defect_types)
        
        # Calculate statistics
        aggregated = {}
        if overall_qualities:
            aggregated['overall_quality'] = np.mean(overall_qualities)
        if dimensional_accuracies:
            aggregated['dimensional_accuracy'] = np.mean(dimensional_accuracies)
        if surface_roughnesses:
            aggregated['surface_roughness'] = np.mean(surface_roughnesses)
        if defect_counts:
            aggregated['defect_count'] = sum(defect_counts)
        if defect_types:
            aggregated['defect_types'] = list(set(defect_types))  # Remove duplicates
        
        return aggregated
    
    def _calculate_fusion_metrics(self, fused_data: FusedVoxelData) -> FusedVoxelData:
        """Calculate fusion confidence and data completeness."""
        # Calculate data completeness
        data_fields = [
            fused_data.ispm_temperature,
            fused_data.ispm_melt_pool_size,
            fused_data.ct_density,
            fused_data.ct_intensity,
            fused_data.overall_quality_score
        ]
        
        non_null_fields = sum(1 for field in data_fields if field is not None)
        completeness = non_null_fields / len(data_fields)
        
        # Calculate fusion confidence based on data quality and consistency
        confidence_factors = []
        
        # ISPM confidence
        if fused_data.ispm_confidence is not None:
            confidence_factors.append(fused_data.ispm_confidence)
        
        # CT quality confidence
        if fused_data.ct_density is not None:
            # Assume higher density indicates better quality
            density_confidence = min(1.0, fused_data.ct_density / 4.43)  # Normalize to theoretical density
            confidence_factors.append(density_confidence)
        
        # Quality score confidence
        if fused_data.overall_quality_score is not None:
            quality_confidence = fused_data.overall_quality_score / 100.0
            confidence_factors.append(quality_confidence)
        
        # Calculate overall confidence
        if confidence_factors:
            fusion_confidence = np.mean(confidence_factors)
        else:
            fusion_confidence = 0.5  # Default confidence
        
        # Update fused data
        fused_data.fusion_confidence = fusion_confidence
        fused_data.data_completeness = completeness
        fused_data.last_updated = datetime.now()
        
        return fused_data
    
    def detect_defects_in_voxels(
        self, 
        fused_data: Dict[Tuple[int, int, int], FusedVoxelData]
    ) -> Dict[Tuple[int, int, int], List[DefectClassification]]:
        """Detect defects in fused voxel data."""
        if not self.config.get('defect_detection_enabled', True):
            return {}
        
        logger.info("Starting defect detection in voxel data...")
        
        defect_map = {}
        
        for voxel_idx, voxel_data in fused_data.items():
            defects = []
            
            # Check for porosity defects
            if voxel_data.ct_porosity is not None and voxel_data.ct_porosity > 0.05:  # 5% porosity threshold
                porosity_defect = DefectClassification(
                    defect_id=f"porosity_{voxel_idx[0]}_{voxel_idx[1]}_{voxel_idx[2]}",
                    defect_type="porosity",
                    severity="medium" if voxel_data.ct_porosity < 0.1 else "high",
                    location="internal",
                    size=voxel_data.ct_porosity * 1000,  # Convert to mm³
                    x_coordinate=voxel_data.voxel_coordinates.x,
                    y_coordinate=voxel_data.voxel_coordinates.y,
                    z_coordinate=voxel_data.voxel_coordinates.z,
                    porosity=voxel_data.ct_porosity,
                    detection_method="ct_scan",
                    detection_confidence=voxel_data.fusion_confidence
                )
                defects.append(porosity_defect)
            
            # Check for quality defects
            if voxel_data.overall_quality_score is not None and voxel_data.overall_quality_score < 80:
                quality_defect = DefectClassification(
                    defect_id=f"quality_{voxel_idx[0]}_{voxel_idx[1]}_{voxel_idx[2]}",
                    defect_type="quality",
                    severity="low" if voxel_data.overall_quality_score > 60 else "medium",
                    location="surface",
                    size=1.0,  # Default size
                    x_coordinate=voxel_data.voxel_coordinates.x,
                    y_coordinate=voxel_data.voxel_coordinates.y,
                    z_coordinate=voxel_data.voxel_coordinates.z,
                    detection_method="quality_metrics",
                    detection_confidence=voxel_data.fusion_confidence
                )
                defects.append(quality_defect)
            
            if defects:
                defect_map[voxel_idx] = defects
                voxel_data.defect_count = len(defects)
                voxel_data.defect_types = [d.defect_type for d in defects]
        
        logger.info(f"Defect detection completed: {len(defect_map)} voxels with defects")
        return defect_map
    
    def export_fused_data(
        self, 
        fused_data: Dict[Tuple[int, int, int], FusedVoxelData],
        output_path: str
    ):
        """Export fused voxel data to file."""
        try:
            import pickle
            
            # Create output directory if it doesn't exist
            from pathlib import Path
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Serialize fused data
            with open(output_path, 'wb') as f:
                pickle.dump(fused_data, f)
            
            logger.info(f"Fused voxel data exported to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting fused data: {e}")
            raise
