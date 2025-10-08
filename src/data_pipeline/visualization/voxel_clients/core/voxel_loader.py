"""
Voxel Data Loader and Parser for PBF-LB/M Visualization

This module provides comprehensive voxel data loading and parsing capabilities
for various data formats used in PBF-LB/M research. It handles loading,
validation, and preprocessing of voxel data from multiple sources.
"""

import numpy as np
import pandas as pd
import h5py
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from datetime import datetime
import zipfile
import tarfile

from src.core.domain.value_objects.voxel_coordinates import VoxelCoordinates
from .cad_voxelizer import VoxelGrid
from .multi_modal_fusion import FusedVoxelData

logger = logging.getLogger(__name__)


@dataclass
class LoadConfig:
    """Configuration for voxel data loading."""
    
    # File format settings
    supported_formats: List[str] = None
    auto_detect_format: bool = True
    validate_data: bool = True
    
    # Memory management
    memory_limit_gb: float = 8.0
    chunk_size: int = 10000
    use_memory_mapping: bool = True
    
    # Data processing
    normalize_coordinates: bool = True
    filter_invalid_voxels: bool = True
    merge_duplicate_voxels: bool = True
    
    # Error handling
    skip_corrupted_files: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class LoadResult:
    """Result of voxel data loading operation."""
    
    success: bool
    data_type: str  # "voxel_grid", "fused_data", "raw_data"
    data: Any
    metadata: Dict[str, Any]
    load_time: float
    file_size: int
    voxel_count: int
    error_message: Optional[str] = None


class VoxelLoader:
    """
    Voxel data loader and parser for PBF-LB/M visualization.
    
    This class provides comprehensive data loading capabilities including:
    - Multiple file format support
    - Data validation and preprocessing
    - Memory-efficient loading
    - Error handling and recovery
    - Metadata extraction
    - Data format conversion
    """
    
    def __init__(self, config: LoadConfig = None):
        """Initialize the voxel loader."""
        self.config = config or LoadConfig()
        if self.config.supported_formats is None:
            self.config.supported_formats = [
                '.npz', '.h5', '.hdf5', '.pkl', '.pickle', '.json', 
                '.csv', '.txt', '.zip', '.tar.gz'
            ]
        
        self.load_cache = {}
        self.format_handlers = self._initialize_format_handlers()
        
        logger.info("Voxel Loader initialized")
    
    def _initialize_format_handlers(self) -> Dict[str, callable]:
        """Initialize format-specific handlers."""
        return {
            '.npz': self._load_npz,
            '.h5': self._load_hdf5,
            '.hdf5': self._load_hdf5,
            '.pkl': self._load_pickle,
            '.pickle': self._load_pickle,
            '.json': self._load_json,
            '.csv': self._load_csv,
            '.txt': self._load_text,
            '.zip': self._load_archive,
            '.tar.gz': self._load_archive
        }
    
    def load_voxel_data(
        self, 
        file_path: Union[str, Path], 
        data_type: str = "auto"
    ) -> LoadResult:
        """
        Load voxel data from file.
        
        Args:
            file_path: Path to the data file
            data_type: Type of data to load ("voxel_grid", "fused_data", "raw_data", "auto")
            
        Returns:
            LoadResult: Loading result with data and metadata
        """
        try:
            start_time = datetime.now()
            file_path = Path(file_path)
            
            # Validate file
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Check file size
            file_size = file_path.stat().st_size
            
            # Detect format
            file_format = self._detect_format(file_path)
            if file_format not in self.format_handlers:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            # Load data
            handler = self.format_handlers[file_format]
            raw_data = handler(file_path)
            
            # Auto-detect data type if needed
            if data_type == "auto":
                data_type = self._detect_data_type(raw_data)
            
            # Process data based on type
            processed_data = self._process_data(raw_data, data_type)
            
            # Validate data
            if self.config.validate_data:
                self._validate_data(processed_data, data_type)
            
            # Calculate load time
            load_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = LoadResult(
                success=True,
                data_type=data_type,
                data=processed_data,
                metadata={
                    'file_path': str(file_path),
                    'file_format': file_format,
                    'file_size': file_size,
                    'load_time': load_time,
                    'data_type': data_type
                },
                load_time=load_time,
                file_size=file_size,
                voxel_count=self._count_voxels(processed_data, data_type)
            )
            
            # Cache result
            self._cache_load_result(result)
            
            logger.info(f"Voxel data loaded successfully: {file_path} ({load_time:.2f}s)")
            return result
            
        except Exception as e:
            logger.error(f"Error loading voxel data from {file_path}: {e}")
            return LoadResult(
                success=False,
                data_type=data_type,
                data=None,
                metadata={'file_path': str(file_path)},
                load_time=0.0,
                file_size=0,
                voxel_count=0,
                error_message=str(e)
            )
    
    def load_voxel_grid(self, file_path: Union[str, Path]) -> VoxelGrid:
        """Load voxel grid from file."""
        result = self.load_voxel_data(file_path, "voxel_grid")
        if not result.success:
            raise ValueError(f"Failed to load voxel grid: {result.error_message}")
        return result.data
    
    def load_fused_data(self, file_path: Union[str, Path]) -> Dict[Tuple[int, int, int], FusedVoxelData]:
        """Load fused voxel data from file."""
        result = self.load_voxel_data(file_path, "fused_data")
        if not result.success:
            raise ValueError(f"Failed to load fused data: {result.error_message}")
        return result.data
    
    def _detect_format(self, file_path: Path) -> str:
        """Detect file format from extension."""
        suffix = file_path.suffix.lower()
        if suffix in self.config.supported_formats:
            return suffix
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def _detect_data_type(self, raw_data: Any) -> str:
        """Auto-detect data type from raw data."""
        if isinstance(raw_data, dict):
            if 'voxels' in raw_data and 'dimensions' in raw_data:
                return "voxel_grid"
            elif any(isinstance(k, tuple) and len(k) == 3 for k in raw_data.keys()):
                return "fused_data"
            else:
                return "raw_data"
        elif isinstance(raw_data, np.ndarray):
            return "raw_data"
        else:
            return "raw_data"
    
    def _process_data(self, raw_data: Any, data_type: str) -> Any:
        """Process raw data based on type."""
        if data_type == "voxel_grid":
            return self._process_voxel_grid(raw_data)
        elif data_type == "fused_data":
            return self._process_fused_data(raw_data)
        else:
            return raw_data
    
    def _process_voxel_grid(self, raw_data: Dict) -> VoxelGrid:
        """Process raw data into VoxelGrid."""
        # Extract required fields
        voxels = raw_data['voxels']
        dimensions = tuple(raw_data['dimensions'])
        voxel_size = float(raw_data['voxel_size'])
        origin = tuple(raw_data['origin'])
        bounds = tuple(raw_data['bounds'])
        
        # Extract optional fields
        material_map = raw_data.get('material_map', np.zeros_like(voxels, dtype=np.uint8))
        process_map = raw_data.get('process_map', {})
        creation_time = datetime.fromisoformat(raw_data.get('creation_time', datetime.now().isoformat()))
        cad_file_path = raw_data.get('cad_file_path', None)
        
        # Create VoxelGrid
        voxel_grid = VoxelGrid(
            origin=origin,
            dimensions=dimensions,
            voxel_size=voxel_size,
            bounds=bounds,
            voxels=voxels,
            material_map=material_map,
            process_map=process_map,
            creation_time=creation_time,
            cad_file_path=cad_file_path
        )
        
        # Calculate statistics
        voxel_grid.total_voxels = np.prod(dimensions)
        voxel_grid.solid_voxels = np.sum(voxels > 0)
        voxel_grid.void_voxels = voxel_grid.total_voxels - voxel_grid.solid_voxels
        
        return voxel_grid
    
    def _process_fused_data(self, raw_data: Dict) -> Dict[Tuple[int, int, int], FusedVoxelData]:
        """Process raw data into fused voxel data."""
        fused_data = {}
        
        for voxel_idx_str, voxel_data_dict in raw_data.items():
            # Convert string key back to tuple
            voxel_idx = eval(voxel_idx_str) if isinstance(voxel_idx_str, str) else voxel_idx_str
            
            # Create FusedVoxelData object
            fused_voxel = FusedVoxelData(
                voxel_coordinates=VoxelCoordinates(
                    x=voxel_data_dict['voxel_coordinates']['x'],
                    y=voxel_data_dict['voxel_coordinates']['y'],
                    z=voxel_data_dict['voxel_coordinates']['z'],
                    voxel_size=voxel_data_dict['voxel_coordinates']['voxel_size'],
                    is_solid=voxel_data_dict['voxel_coordinates']['is_solid']
                ),
                is_solid=voxel_data_dict['is_solid'],
                material_type=voxel_data_dict['material_type'],
                laser_power=voxel_data_dict['laser_power'],
                scan_speed=voxel_data_dict['scan_speed'],
                layer_number=voxel_data_dict['layer_number'],
                build_time=datetime.fromisoformat(voxel_data_dict['build_time']),
                ispm_temperature=voxel_data_dict.get('ispm_temperature'),
                ispm_melt_pool_size=voxel_data_dict.get('ispm_melt_pool_size'),
                ispm_acoustic_emissions=voxel_data_dict.get('ispm_acoustic_emissions'),
                ispm_plume_intensity=voxel_data_dict.get('ispm_plume_intensity'),
                ispm_confidence=voxel_data_dict.get('ispm_confidence'),
                ct_density=voxel_data_dict.get('ct_density'),
                ct_intensity=voxel_data_dict.get('ct_intensity'),
                ct_defect_probability=voxel_data_dict.get('ct_defect_probability'),
                ct_porosity=voxel_data_dict.get('ct_porosity'),
                overall_quality_score=voxel_data_dict.get('overall_quality_score', 100.0),
                dimensional_accuracy=voxel_data_dict.get('dimensional_accuracy'),
                surface_roughness=voxel_data_dict.get('surface_roughness'),
                defect_count=voxel_data_dict.get('defect_count', 0),
                defect_types=voxel_data_dict.get('defect_types', []),
                fusion_confidence=voxel_data_dict.get('fusion_confidence', 1.0),
                data_completeness=voxel_data_dict.get('data_completeness', 1.0),
                last_updated=datetime.fromisoformat(voxel_data_dict.get('last_updated', datetime.now().isoformat()))
            )
            
            fused_data[voxel_idx] = fused_voxel
        
        return fused_data
    
    def _validate_data(self, data: Any, data_type: str):
        """Validate loaded data."""
        if data_type == "voxel_grid":
            self._validate_voxel_grid(data)
        elif data_type == "fused_data":
            self._validate_fused_data(data)
    
    def _validate_voxel_grid(self, voxel_grid: VoxelGrid):
        """Validate voxel grid data."""
        if not isinstance(voxel_grid, VoxelGrid):
            raise ValueError("Invalid voxel grid type")
        
        if voxel_grid.voxels is None:
            raise ValueError("Voxel data is None")
        
        if voxel_grid.dimensions != voxel_grid.voxels.shape:
            raise ValueError("Dimension mismatch")
        
        if voxel_grid.voxel_size <= 0:
            raise ValueError("Invalid voxel size")
    
    def _validate_fused_data(self, fused_data: Dict):
        """Validate fused voxel data."""
        if not isinstance(fused_data, dict):
            raise ValueError("Invalid fused data type")
        
        for voxel_idx, voxel_data in fused_data.items():
            if not isinstance(voxel_idx, tuple) or len(voxel_idx) != 3:
                raise ValueError(f"Invalid voxel index: {voxel_idx}")
            
            if not isinstance(voxel_data, FusedVoxelData):
                raise ValueError(f"Invalid voxel data type for {voxel_idx}")
    
    def _count_voxels(self, data: Any, data_type: str) -> int:
        """Count voxels in data."""
        if data_type == "voxel_grid":
            return data.solid_voxels
        elif data_type == "fused_data":
            return len(data)
        else:
            return 0
    
    def _load_npz(self, file_path: Path) -> Dict:
        """Load NumPy compressed format."""
        return dict(np.load(file_path, allow_pickle=True))
    
    def _load_hdf5(self, file_path: Path) -> Dict:
        """Load HDF5 format."""
        data = {}
        with h5py.File(file_path, 'r') as f:
            def extract_data(name, obj):
                if isinstance(obj, h5py.Dataset):
                    data[name] = obj[:]
                elif isinstance(obj, h5py.Group):
                    data[name] = {}
                    for key in obj.keys():
                        data[name][key] = obj[key][:]
            
            f.visititems(extract_data)
        
        return data
    
    def _load_pickle(self, file_path: Path) -> Any:
        """Load pickle format."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def _load_json(self, file_path: Path) -> Dict:
        """Load JSON format."""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def _load_csv(self, file_path: Path) -> pd.DataFrame:
        """Load CSV format."""
        return pd.read_csv(file_path)
    
    def _load_text(self, file_path: Path) -> str:
        """Load text format."""
        with open(file_path, 'r') as f:
            return f.read()
    
    def _load_archive(self, file_path: Path) -> Dict:
        """Load archive format (ZIP, TAR.GZ)."""
        data = {}
        
        if file_path.suffix == '.zip':
            with zipfile.ZipFile(file_path, 'r') as zip_file:
                for file_info in zip_file.filelist:
                    if not file_info.is_dir():
                        file_data = zip_file.read(file_info.filename)
                        data[file_info.filename] = file_data
        
        elif file_path.suffix == '.gz' and file_path.name.endswith('.tar.gz'):
            with tarfile.open(file_path, 'r:gz') as tar_file:
                for member in tar_file.getmembers():
                    if member.isfile():
                        file_data = tar_file.extractfile(member)
                        if file_data:
                            data[member.name] = file_data.read()
        
        return data
    
    def _cache_load_result(self, result: LoadResult):
        """Cache load result for performance."""
        cache_key = f"{result.metadata['file_path']}_{result.data_type}"
        self.load_cache[cache_key] = result
    
    def get_cached_result(self, file_path: Union[str, Path], data_type: str) -> Optional[LoadResult]:
        """Get cached load result."""
        cache_key = f"{file_path}_{data_type}"
        return self.load_cache.get(cache_key)
    
    def clear_cache(self):
        """Clear load cache."""
        self.load_cache.clear()
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return self.config.supported_formats.copy()
    
    def get_load_statistics(self) -> Dict[str, Any]:
        """Get loading statistics."""
        return {
            'cache_size': len(self.load_cache),
            'supported_formats': self.config.supported_formats,
            'memory_limit_gb': self.config.memory_limit_gb,
            'chunk_size': self.config.chunk_size
        }
