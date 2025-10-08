"""
Voxel coordinates value object for PBF-LB/M operations.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import math

from .base_value_object import BaseValueObject


@dataclass(frozen=True)
class VoxelCoordinates(BaseValueObject):
    """
    Value object representing voxel coordinates for PBF-LB/M operations.
    
    This immutable object contains 3D coordinate information for voxels
    in PBF processes, including position, orientation, and metadata.
    """
    
    # Basic coordinates
    x: float  # mm
    y: float  # mm
    z: float  # mm
    
    # Voxel properties
    voxel_size: float = 0.1  # mm
    voxel_volume: Optional[float] = None  # mm³
    
    # Orientation (Euler angles in degrees)
    rotation_x: float = 0.0  # degrees
    rotation_y: float = 0.0  # degrees
    rotation_z: float = 0.0  # degrees
    
    # Voxel state
    is_solid: bool = True
    is_processed: bool = False
    is_defective: bool = False
    
    # Material properties
    material_density: Optional[float] = None  # g/cm³
    material_type: Optional[str] = None
    
    # Process information
    layer_number: Optional[int] = None
    scan_vector_id: Optional[str] = None
    processing_timestamp: Optional[datetime] = None
    
    # Quality metrics
    quality_score: Optional[float] = None  # 0-100
    temperature_peak: Optional[float] = None  # Celsius
    cooling_rate: Optional[float] = None  # K/s
    
    def __post_init__(self):
        """Calculate derived properties and validate."""
        if self.voxel_volume is None:
            object.__setattr__(self, 'voxel_volume', self.voxel_size ** 3)
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate voxel coordinates."""
        # Coordinate validation
        if self.x < 0 or self.y < 0 or self.z < 0:
            raise ValueError("Coordinates cannot be negative")
        
        # Voxel size validation
        if self.voxel_size <= 0:
            raise ValueError("Voxel size must be positive")
        
        # Rotation validation
        for rotation in [self.rotation_x, self.rotation_y, self.rotation_z]:
            if not -180 <= rotation <= 180:
                raise ValueError("Rotation angles must be between -180 and 180 degrees")
        
        # Material density validation
        if self.material_density is not None and self.material_density <= 0:
            raise ValueError("Material density must be positive")
        
        # Quality score validation
        if self.quality_score is not None and not 0 <= self.quality_score <= 100:
            raise ValueError("Quality score must be between 0 and 100")
        
        # Temperature validation
        if self.temperature_peak is not None and self.temperature_peak < 0:
            raise ValueError("Temperature cannot be negative")
        
        # Cooling rate validation
        if self.cooling_rate is not None and self.cooling_rate < 0:
            raise ValueError("Cooling rate cannot be negative")
    
    def get_coordinates(self) -> Tuple[float, float, float]:
        """Get coordinates as tuple."""
        return (self.x, self.y, self.z)
    
    def get_rotations(self) -> Tuple[float, float, float]:
        """Get rotations as tuple."""
        return (self.rotation_x, self.rotation_y, self.rotation_z)
    
    def distance_to(self, other: 'VoxelCoordinates') -> float:
        """Calculate Euclidean distance to another voxel."""
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def distance_to_origin(self) -> float:
        """Calculate distance from origin."""
        return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
    
    def is_adjacent_to(self, other: 'VoxelCoordinates', tolerance: float = 0.1) -> bool:
        """Check if this voxel is adjacent to another voxel."""
        distance = self.distance_to(other)
        return distance <= (self.voxel_size + other.voxel_size) / 2 + tolerance
    
    def is_in_bounds(self, bounds: Dict[str, float]) -> bool:
        """Check if voxel is within specified bounds."""
        return (bounds.get('min_x', 0) <= self.x <= bounds.get('max_x', float('inf')) and
                bounds.get('min_y', 0) <= self.y <= bounds.get('max_y', float('inf')) and
                bounds.get('min_z', 0) <= self.z <= bounds.get('max_z', float('inf')))
    
    def get_bounding_box(self) -> Dict[str, float]:
        """Get bounding box for this voxel."""
        half_size = self.voxel_size / 2
        return {
            'min_x': self.x - half_size,
            'max_x': self.x + half_size,
            'min_y': self.y - half_size,
            'max_y': self.y + half_size,
            'min_z': self.z - half_size,
            'max_z': self.z + half_size
        }
    
    def get_center_point(self) -> Tuple[float, float, float]:
        """Get center point of the voxel."""
        return (self.x, self.y, self.z)
    
    def get_corners(self) -> List[Tuple[float, float, float]]:
        """Get all 8 corners of the voxel."""
        half_size = self.voxel_size / 2
        corners = []
        
        for dx in [-half_size, half_size]:
            for dy in [-half_size, half_size]:
                for dz in [-half_size, half_size]:
                    corners.append((self.x + dx, self.y + dy, self.z + dz))
        
        return corners
    
    def get_face_centers(self) -> List[Tuple[float, float, float]]:
        """Get center points of all 6 faces."""
        half_size = self.voxel_size / 2
        faces = [
            (self.x + half_size, self.y, self.z),      # +X face
            (self.x - half_size, self.y, self.z),      # -X face
            (self.x, self.y + half_size, self.z),      # +Y face
            (self.x, self.y - half_size, self.z),      # -Y face
            (self.x, self.y, self.z + half_size),      # +Z face
            (self.x, self.y, self.z - half_size),      # -Z face
        ]
        return faces
    
    def get_volume(self) -> float:
        """Get volume of the voxel."""
        return self.voxel_volume
    
    def get_mass(self) -> Optional[float]:
        """Get mass of the voxel if material density is known."""
        if self.material_density is None:
            return None
        
        # Convert density from g/cm³ to g/mm³
        density_mm3 = self.material_density / 1000
        return self.voxel_volume * density_mm3
    
    def get_orientation_vector(self) -> Tuple[float, float, float]:
        """Get orientation as unit vector."""
        # Convert Euler angles to radians
        rx = math.radians(self.rotation_x)
        ry = math.radians(self.rotation_y)
        rz = math.radians(self.rotation_z)
        
        # Calculate rotation matrix components
        cos_x, sin_x = math.cos(rx), math.sin(rx)
        cos_y, sin_y = math.cos(ry), math.sin(ry)
        cos_z, sin_z = math.cos(rz), math.sin(rz)
        
        # Z-axis vector after rotation
        zx = -sin_y
        zy = sin_x * cos_y
        zz = cos_x * cos_y
        
        return (zx, zy, zz)
    
    def is_in_same_layer(self, other: 'VoxelCoordinates', tolerance: float = 0.1) -> bool:
        """Check if two voxels are in the same layer."""
        return abs(self.z - other.z) <= tolerance
    
    def get_layer_position(self) -> int:
        """Get layer position based on Z coordinate."""
        if self.layer_number is not None:
            return self.layer_number
        return int(self.z / self.voxel_size)
    
    def get_scan_order(self, scan_pattern: str = "zigzag") -> int:
        """Get scan order within layer based on pattern."""
        if scan_pattern == "zigzag":
            # Simple zigzag pattern
            return int(self.x + self.y * 1000)
        elif scan_pattern == "spiral":
            # Spiral pattern from center
            center_x, center_y = 0, 0  # Assuming center at origin
            distance = math.sqrt((self.x - center_x)**2 + (self.y - center_y)**2)
            angle = math.atan2(self.y - center_y, self.x - center_x)
            return int(distance * 1000 + angle * 100)
        else:
            # Default: row by row
            return int(self.y * 1000 + self.x)
    
    def get_neighbor_coordinates(self, include_diagonals: bool = False) -> List['VoxelCoordinates']:
        """Get coordinates of neighboring voxels."""
        neighbors = []
        step = self.voxel_size
        
        # Face neighbors (6 directions)
        directions = [
            (step, 0, 0), (-step, 0, 0),
            (0, step, 0), (0, -step, 0),
            (0, 0, step), (0, 0, -step)
        ]
        
        if include_diagonals:
            # Add diagonal neighbors (20 additional directions)
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    for dz in [-step, 0, step]:
                        if dx != 0 or dy != 0 or dz != 0:
                            directions.append((dx, dy, dz))
        
        for dx, dy, dz in directions:
            neighbor = VoxelCoordinates(
                x=self.x + dx,
                y=self.y + dy,
                z=self.z + dz,
                voxel_size=self.voxel_size,
                created_at=self.created_at,
                updated_at=self.updated_at
            )
            neighbors.append(neighbor)
        
        return neighbors
    
    def get_voxel_info(self) -> Dict[str, Any]:
        """Get comprehensive voxel information."""
        return {
            "coordinates": self.get_coordinates(),
            "voxel_size": self.voxel_size,
            "volume": self.get_volume(),
            "mass": self.get_mass(),
            "rotations": self.get_rotations(),
            "orientation_vector": self.get_orientation_vector(),
            "is_solid": self.is_solid,
            "is_processed": self.is_processed,
            "is_defective": self.is_defective,
            "material_type": self.material_type,
            "material_density": self.material_density,
            "layer_number": self.get_layer_position(),
            "scan_vector_id": self.scan_vector_id,
            "quality_score": self.quality_score,
            "temperature_peak": self.temperature_peak,
            "cooling_rate": self.cooling_rate,
            "bounding_box": self.get_bounding_box(),
            "distance_to_origin": self.distance_to_origin()
        }
    
    def to_voxel_grid_position(self, grid_origin: Tuple[float, float, float] = (0, 0, 0)) -> Tuple[int, int, int]:
        """Convert to voxel grid position indices."""
        grid_x = int((self.x - grid_origin[0]) / self.voxel_size)
        grid_y = int((self.y - grid_origin[1]) / self.voxel_size)
        grid_z = int((self.z - grid_origin[2]) / self.voxel_size)
        return (grid_x, grid_y, grid_z)
    
    @classmethod
    def from_voxel_grid_position(cls, grid_x: int, grid_y: int, grid_z: int, 
                                voxel_size: float = 0.1,
                                grid_origin: Tuple[float, float, float] = (0, 0, 0)) -> 'VoxelCoordinates':
        """Create voxel coordinates from grid position."""
        x = grid_origin[0] + grid_x * voxel_size
        y = grid_origin[1] + grid_y * voxel_size
        z = grid_origin[2] + grid_z * voxel_size
        
        return cls(
            x=x, y=y, z=z,
            voxel_size=voxel_size,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )