"""
CAD Voxelization System for PBF-LB/M Models

This module provides comprehensive CAD model voxelization capabilities for PBF-LB/M
(Powder Bed Fusion - Laser Beam/Metal) additive manufacturing research. It converts
CAD models into voxel representations that can be integrated with process parameters,
ISPM data, and CT scan data for spatially-resolved analysis.
"""

import numpy as np
import trimesh
import pyvista as pv
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime

from src.core.domain.value_objects.voxel_coordinates import VoxelCoordinates
from src.core.domain.entities.pbf_process import PBFProcess
from src.core.domain.value_objects.process_parameters import ProcessParameters

logger = logging.getLogger(__name__)


@dataclass
class VoxelGrid:
    """Voxel grid representation for PBF-LB/M models."""
    
    # Grid properties
    origin: Tuple[float, float, float]  # (x, y, z) origin in mm
    dimensions: Tuple[int, int, int]    # (width, height, depth) in voxels
    voxel_size: float                   # Voxel size in mm
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]  # ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    
    # Voxel data
    voxels: np.ndarray                  # 3D array of voxel data
    material_map: np.ndarray            # Material type for each voxel
    process_map: Dict[str, np.ndarray]  # Process parameters mapped to voxels
    
    # Metadata
    creation_time: datetime
    cad_file_path: Optional[str] = None
    total_voxels: int = 0
    solid_voxels: int = 0
    void_voxels: int = 0


@dataclass
class VoxelizationConfig:
    """Configuration for CAD voxelization."""
    
    voxel_size: float = 0.1  # mm
    material_type: str = "Ti-6Al-4V"
    coordinate_system: str = "right_handed"  # or "left_handed"
    precision: int = 6  # decimal places for coordinates
    memory_limit_gb: float = 8.0
    parallel_processing: bool = True
    num_workers: int = 4


class CADVoxelizer:
    """
    CAD model voxelization system for PBF-LB/M research.
    
    This class provides comprehensive voxelization capabilities including:
    - CAD model loading and parsing
    - Voxel grid generation
    - Geometry rasterization
    - Process parameter registration
    - Multi-modal data fusion
    """
    
    def __init__(self, config: VoxelizationConfig = None):
        """Initialize the CAD voxelizer."""
        self.config = config or VoxelizationConfig()
        self.supported_formats = ['.stl', '.step', '.iges', '.obj', '.ply', '.3mf']
        self.voxel_cache = {}
        
        logger.info(f"CAD Voxelizer initialized with voxel size: {self.config.voxel_size}mm")
    
    def voxelize_cad_model(
        self, 
        cad_file_path: Union[str, Path], 
        process_parameters: Optional[ProcessParameters] = None
    ) -> VoxelGrid:
        """
        Voxelize a CAD model for PBF-LB/M analysis.
        
        Args:
            cad_file_path: Path to CAD model file
            process_parameters: PBF-LB/M process parameters for registration
            
        Returns:
            VoxelGrid: Voxelized representation of the CAD model
        """
        try:
            logger.info(f"Starting voxelization of CAD model: {cad_file_path}")
            
            # Load and validate CAD model
            mesh = self._load_cad_model(cad_file_path)
            
            # Calculate voxel grid bounds and dimensions
            bounds, dimensions = self._calculate_voxel_grid(mesh)
            
            # Create voxel grid
            voxel_grid = self._create_voxel_grid(bounds, dimensions, str(cad_file_path))
            
            # Rasterize geometry
            self._rasterize_geometry(mesh, voxel_grid)
            
            # Register process parameters if provided
            if process_parameters:
                self._register_process_parameters(voxel_grid, process_parameters)
            
            # Calculate statistics
            self._calculate_voxel_statistics(voxel_grid)
            
            logger.info(f"Voxelization completed: {voxel_grid.total_voxels} total voxels, "
                       f"{voxel_grid.solid_voxels} solid voxels")
            
            return voxel_grid
            
        except Exception as e:
            logger.error(f"Error voxelizing CAD model {cad_file_path}: {e}")
            raise
    
    def _load_cad_model(self, file_path: Union[str, Path]) -> trimesh.Trimesh:
        """Load CAD model from file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"CAD file not found: {file_path}")
        
        if file_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        try:
            # Load mesh using trimesh
            mesh = trimesh.load(str(file_path))
            
            # Ensure mesh is watertight for proper voxelization
            if not mesh.is_watertight:
                logger.warning(f"Mesh is not watertight, attempting repair...")
                mesh.fill_holes()
                mesh.remove_duplicate_faces()
                mesh.remove_degenerate_faces()
            
            # Validate mesh
            if not mesh.is_valid:
                raise ValueError("Invalid mesh geometry")
            
            logger.info(f"Loaded CAD model: {len(mesh.vertices)} vertices, "
                       f"{len(mesh.faces)} faces")
            
            return mesh
            
        except Exception as e:
            logger.error(f"Error loading CAD model: {e}")
            raise
    
    def _calculate_voxel_grid(self, mesh: trimesh.Trimesh) -> Tuple[Tuple, Tuple]:
        """Calculate voxel grid bounds and dimensions."""
        # Get mesh bounds
        bounds = mesh.bounds  # ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        
        # Calculate dimensions in voxels
        dimensions = tuple(
            int(np.ceil((bounds[i][1] - bounds[i][0]) / self.config.voxel_size))
            for i in range(3)
        )
        
        # Check memory requirements
        total_voxels = np.prod(dimensions)
        memory_required_gb = (total_voxels * 8) / (1024**3)  # 8 bytes per voxel
        
        if memory_required_gb > self.config.memory_limit_gb:
            raise MemoryError(f"Voxel grid too large: {memory_required_gb:.2f}GB required, "
                            f"{self.config.memory_limit_gb}GB limit")
        
        logger.info(f"Voxel grid dimensions: {dimensions} ({total_voxels:,} voxels)")
        logger.info(f"Memory required: {memory_required_gb:.2f}GB")
        
        return bounds, dimensions
    
    def _create_voxel_grid(
        self, 
        bounds: Tuple, 
        dimensions: Tuple, 
        cad_file_path: str
    ) -> VoxelGrid:
        """Create voxel grid structure."""
        # Calculate origin (minimum bounds)
        origin = (bounds[0][0], bounds[1][0], bounds[2][0])
        
        # Initialize voxel arrays
        voxels = np.zeros(dimensions, dtype=np.float32)
        material_map = np.zeros(dimensions, dtype=np.uint8)
        
        # Create process parameter maps
        process_map = {
            'laser_power': np.zeros(dimensions, dtype=np.float32),
            'scan_speed': np.zeros(dimensions, dtype=np.float32),
            'layer_number': np.zeros(dimensions, dtype=np.int16),
            'build_time': np.zeros(dimensions, dtype=np.float64),
            'temperature': np.zeros(dimensions, dtype=np.float32),
            'quality_score': np.zeros(dimensions, dtype=np.float32)
        }
        
        voxel_grid = VoxelGrid(
            origin=origin,
            dimensions=dimensions,
            voxel_size=self.config.voxel_size,
            bounds=bounds,
            voxels=voxels,
            material_map=material_map,
            process_map=process_map,
            creation_time=datetime.now(),
            cad_file_path=cad_file_path
        )
        
        return voxel_grid
    
    def _rasterize_geometry(self, mesh: trimesh.Trimesh, voxel_grid: VoxelGrid):
        """Rasterize mesh geometry into voxel grid."""
        logger.info("Starting geometry rasterization...")
        
        # Create voxel grid using trimesh voxelization
        voxelized = mesh.voxelized(pitch=self.config.voxel_size)
        
        # Get voxel coordinates
        voxel_coords = voxelized.points
        
        # Convert to grid indices
        grid_indices = self._world_to_grid_coordinates(
            voxel_coords, voxel_grid.origin, voxel_grid.voxel_size
        )
        
        # Set solid voxels
        for idx in grid_indices:
            if self._is_valid_grid_index(idx, voxel_grid.dimensions):
                voxel_grid.voxels[idx] = 1.0
                voxel_grid.material_map[idx] = 1  # Default material ID
        
        logger.info(f"Rasterized {len(grid_indices)} solid voxels")
    
    def _world_to_grid_coordinates(
        self, 
        world_coords: np.ndarray, 
        origin: Tuple[float, float, float], 
        voxel_size: float
    ) -> np.ndarray:
        """Convert world coordinates to grid indices."""
        grid_coords = np.floor((world_coords - np.array(origin)) / voxel_size).astype(int)
        return grid_coords
    
    def _is_valid_grid_index(self, index: np.ndarray, dimensions: Tuple[int, int, int]) -> bool:
        """Check if grid index is valid."""
        return (0 <= index[0] < dimensions[0] and 
                0 <= index[1] < dimensions[1] and 
                0 <= index[2] < dimensions[2])
    
    def _register_process_parameters(self, voxel_grid: VoxelGrid, process_params: ProcessParameters):
        """Register PBF-LB/M process parameters to voxel coordinates."""
        logger.info("Registering process parameters to voxel coordinates...")
        
        # Map laser scan path to voxels
        if hasattr(process_params, 'laser_scan_path'):
            self._map_laser_scan_path(voxel_grid, process_params.laser_scan_path)
        
        # Map layer information
        if hasattr(process_params, 'layers'):
            self._map_layer_information(voxel_grid, process_params.layers)
        
        # Map build parameters
        self._map_build_parameters(voxel_grid, process_params)
        
        logger.info("Process parameter registration completed")
    
    def _map_laser_scan_path(self, voxel_grid: VoxelGrid, scan_path: List[Dict]):
        """Map laser scan path to voxel coordinates."""
        for scan_segment in scan_path:
            start_point = scan_segment['start']
            end_point = scan_segment['end']
            laser_power = scan_segment.get('laser_power', 0.0)
            scan_speed = scan_segment.get('scan_speed', 0.0)
            
            # Get voxels along scan path
            path_voxels = self._get_voxels_along_path(
                start_point, end_point, voxel_grid
            )
            
            # Set process parameters for path voxels
            for voxel_idx in path_voxels:
                voxel_grid.process_map['laser_power'][voxel_idx] = laser_power
                voxel_grid.process_map['scan_speed'][voxel_idx] = scan_speed
    
    def _map_layer_information(self, voxel_grid: VoxelGrid, layers: List[Dict]):
        """Map layer information to voxel coordinates."""
        for layer_idx, layer_data in enumerate(layers):
            layer_height = layer_data.get('height', 0.0)
            layer_time = layer_data.get('build_time', 0.0)
            
            # Find voxels in this layer
            layer_voxels = self._get_voxels_in_layer(
                layer_height, voxel_grid
            )
            
            # Set layer parameters
            for voxel_idx in layer_voxels:
                voxel_grid.process_map['layer_number'][voxel_idx] = layer_idx
                voxel_grid.process_map['build_time'][voxel_idx] = layer_time
    
    def _map_build_parameters(self, voxel_grid: VoxelGrid, process_params: ProcessParameters):
        """Map general build parameters to voxel coordinates."""
        # Set default values for all solid voxels
        solid_voxels = np.where(voxel_grid.voxels > 0)
        
        for i in range(len(solid_voxels[0])):
            idx = (solid_voxels[0][i], solid_voxels[1][i], solid_voxels[2][i])
            
            # Set default process parameters
            voxel_grid.process_map['laser_power'][idx] = getattr(process_params, 'laser_power', 0.0)
            voxel_grid.process_map['scan_speed'][idx] = getattr(process_params, 'scan_speed', 0.0)
            voxel_grid.process_map['temperature'][idx] = getattr(process_params, 'temperature', 0.0)
            voxel_grid.process_map['quality_score'][idx] = 100.0  # Default quality score
    
    def _get_voxels_along_path(
        self, 
        start_point: Tuple[float, float, float], 
        end_point: Tuple[float, float, float], 
        voxel_grid: VoxelGrid
    ) -> List[Tuple[int, int, int]]:
        """Get voxel indices along a path between two points."""
        # Convert world coordinates to grid coordinates
        start_grid = self._world_to_grid_coordinates(
            np.array(start_point), voxel_grid.origin, voxel_grid.voxel_size
        )
        end_grid = self._world_to_grid_coordinates(
            np.array(end_point), voxel_grid.origin, voxel_grid.voxel_size
        )
        
        # Generate voxels along path (simplified line algorithm)
        path_voxels = []
        steps = max(abs(end_grid[0] - start_grid[0]), 
                   abs(end_grid[1] - start_grid[1]), 
                   abs(end_grid[2] - start_grid[2]))
        
        for i in range(steps + 1):
            t = i / steps if steps > 0 else 0
            point = start_grid + t * (end_grid - start_grid)
            point = point.astype(int)
            
            if self._is_valid_grid_index(point, voxel_grid.dimensions):
                path_voxels.append(tuple(point))
        
        return path_voxels
    
    def _get_voxels_in_layer(
        self, 
        layer_height: float, 
        voxel_grid: VoxelGrid
    ) -> List[Tuple[int, int, int]]:
        """Get voxel indices in a specific layer."""
        # Calculate layer bounds in grid coordinates
        layer_min = layer_height - voxel_grid.voxel_size / 2
        layer_max = layer_height + voxel_grid.voxel_size / 2
        
        # Convert to grid coordinates
        min_grid_z = int((layer_min - voxel_grid.origin[2]) / voxel_grid.voxel_size)
        max_grid_z = int((layer_max - voxel_grid.origin[2]) / voxel_grid.voxel_size)
        
        # Find voxels in layer
        layer_voxels = []
        for z in range(max(0, min_grid_z), min(voxel_grid.dimensions[2], max_grid_z + 1)):
            solid_voxels = np.where(voxel_grid.voxels[:, :, z] > 0)
            for i in range(len(solid_voxels[0])):
                layer_voxels.append((solid_voxels[0][i], solid_voxels[1][i], z))
        
        return layer_voxels
    
    def _calculate_voxel_statistics(self, voxel_grid: VoxelGrid):
        """Calculate voxel grid statistics."""
        voxel_grid.total_voxels = np.prod(voxel_grid.dimensions)
        voxel_grid.solid_voxels = np.sum(voxel_grid.voxels > 0)
        voxel_grid.void_voxels = voxel_grid.total_voxels - voxel_grid.solid_voxels
        
        logger.info(f"Voxel statistics: {voxel_grid.solid_voxels:,} solid, "
                   f"{voxel_grid.void_voxels:,} void, "
                   f"{voxel_grid.solid_voxels/voxel_grid.total_voxels*100:.1f}% fill ratio")
    
    def get_voxel_coordinates(
        self, 
        voxel_grid: VoxelGrid, 
        grid_index: Tuple[int, int, int]
    ) -> VoxelCoordinates:
        """Get VoxelCoordinates object for a grid index."""
        # Convert grid index to world coordinates
        world_coords = (
            voxel_grid.origin[0] + grid_index[0] * voxel_grid.voxel_size,
            voxel_grid.origin[1] + grid_index[1] * voxel_grid.voxel_size,
            voxel_grid.origin[2] + grid_index[2] * voxel_grid.voxel_size
        )
        
        # Get process parameters for this voxel
        laser_power = voxel_grid.process_map['laser_power'][grid_index]
        scan_speed = voxel_grid.process_map['scan_speed'][grid_index]
        layer_number = voxel_grid.process_map['layer_number'][grid_index]
        temperature = voxel_grid.process_map['temperature'][grid_index]
        quality_score = voxel_grid.process_map['quality_score'][grid_index]
        
        return VoxelCoordinates(
            x=world_coords[0],
            y=world_coords[1],
            z=world_coords[2],
            voxel_size=voxel_grid.voxel_size,
            is_solid=voxel_grid.voxels[grid_index] > 0,
            layer_number=int(layer_number) if layer_number > 0 else None,
            material_type=self.config.material_type,
            quality_score=float(quality_score),
            temperature_peak=float(temperature)
        )
    
    def export_voxel_grid(self, voxel_grid: VoxelGrid, output_path: str):
        """Export voxel grid to file."""
        try:
            # Create output directory if it doesn't exist
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save as NumPy compressed format
            np.savez_compressed(
                output_path,
                voxels=voxel_grid.voxels,
                material_map=voxel_grid.material_map,
                process_map=voxel_grid.process_map,
                origin=voxel_grid.origin,
                dimensions=voxel_grid.dimensions,
                voxel_size=voxel_grid.voxel_size,
                bounds=voxel_grid.bounds,
                creation_time=voxel_grid.creation_time.isoformat(),
                cad_file_path=voxel_grid.cad_file_path
            )
            
            logger.info(f"Voxel grid exported to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting voxel grid: {e}")
            raise
    
    def load_voxel_grid(self, file_path: str) -> VoxelGrid:
        """Load voxel grid from file."""
        try:
            data = np.load(file_path, allow_pickle=True)
            
            voxel_grid = VoxelGrid(
                origin=tuple(data['origin']),
                dimensions=tuple(data['dimensions']),
                voxel_size=float(data['voxel_size']),
                bounds=tuple(data['bounds']),
                voxels=data['voxels'],
                material_map=data['material_map'],
                process_map=data['process_map'].item(),
                creation_time=datetime.fromisoformat(data['creation_time']),
                cad_file_path=str(data['cad_file_path'])
            )
            
            # Calculate statistics
            self._calculate_voxel_statistics(voxel_grid)
            
            logger.info(f"Voxel grid loaded from: {file_path}")
            return voxel_grid
            
        except Exception as e:
            logger.error(f"Error loading voxel grid: {e}")
            raise
