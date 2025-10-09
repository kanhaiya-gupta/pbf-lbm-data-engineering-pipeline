"""
Core Voxel Renderer for PBF-LB/M Visualization

This module provides the main voxel rendering engine that enables 3D visualization
of voxel data for PBF-LB/M research. It supports multiple rendering techniques
and provides a unified interface for voxel visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pyvista as pv
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from datetime import datetime
import json

from src.core.domain.value_objects.voxel_coordinates import VoxelCoordinates
from .cad_voxelizer import VoxelGrid
from .multi_modal_fusion import FusedVoxelData

logger = logging.getLogger(__name__)


@dataclass
class RenderConfig:
    """Configuration for voxel rendering."""
    
    # Rendering parameters
    render_engine: str = "pyvista"  # "plotly", "matplotlib", "pyvista"
    voxel_size: float = 0.1  # mm
    color_scheme: str = "viridis"  # "viridis", "plasma", "inferno", "magma", "custom"
    opacity: float = 0.8
    show_axes: bool = True
    show_grid: bool = True
    
    # Quality visualization
    quality_colormap: str = "RdYlGn"  # Red-Yellow-Green for quality
    defect_colormap: str = "Reds"  # Red for defects
    process_colormap: str = "Blues"  # Blue for process parameters
    
    # Performance settings
    max_voxels_display: int = 100000  # Maximum voxels to display
    level_of_detail: int = 1  # 1=full, 2=half, 4=quarter resolution
    enable_caching: bool = True
    
    # Export settings
    image_dpi: int = 300
    image_format: str = "png"  # "png", "jpg", "svg", "pdf"
    video_fps: int = 30
    video_format: str = "mp4"  # "mp4", "gif", "avi"


@dataclass
class RenderResult:
    """Result of voxel rendering operation."""
    
    success: bool
    render_type: str  # "3d", "2d", "cross_section", "animation"
    data: Any  # Rendered data (figure, image, etc.)
    metadata: Dict[str, Any]
    render_time: float
    voxel_count: int
    error_message: Optional[str] = None


class VoxelRenderer:
    """
    Main voxel rendering engine for PBF-LB/M visualization.
    
    This class provides comprehensive voxel rendering capabilities including:
    - 3D volume rendering
    - Cross-section visualization
    - Quality-based color mapping
    - Defect highlighting
    - Process parameter visualization
    - Interactive controls
    - Export capabilities
    """
    
    def __init__(self, config: RenderConfig = None):
        """Initialize the voxel renderer."""
        self.config = config or RenderConfig()
        self.render_cache = {}
        self.color_maps = self._initialize_color_maps()
        
        logger.info(f"Voxel Renderer initialized with {self.config.render_engine} engine")
    
    def _initialize_color_maps(self) -> Dict[str, str]:
        """Initialize color map configurations."""
        return {
            'quality': self.config.quality_colormap,
            'defect': self.config.defect_colormap,
            'process': self.config.process_colormap,
            'default': self.config.color_scheme
        }
    
    def render_voxel_grid(
        self,
        voxel_grid: VoxelGrid,
        render_type: str = "3d",
        color_by: str = "quality",
        granularity: str = "per_hatch",  # NEW: per_hatch, per_layer, per_build, custom
        parameters: List[str] = None,    # NEW: List of parameters to visualize
        visualization_mode: str = "interactive",  # NEW: interactive, static, comparison
        **kwargs
    ) -> RenderResult:
        """
        Render a voxel grid with specified parameters.
        
        Args:
            voxel_grid: Voxel grid to render
            render_type: Type of rendering ("3d", "2d", "cross_section")
            color_by: Color mapping ("quality", "defect", "process", "material")
            **kwargs: Additional rendering parameters
            
        Returns:
            RenderResult: Rendering result with data and metadata
        """
        try:
            start_time = datetime.now()
            
            # Validate inputs
            if not self._validate_voxel_grid(voxel_grid):
                return RenderResult(
                    success=False,
                    render_type=render_type,
                    data=None,
                    metadata={},
                    render_time=0.0,
                    voxel_count=0,
                    error_message="Invalid voxel grid"
                )
            
            # Apply level of detail
            processed_grid = self._apply_level_of_detail(voxel_grid)
            
            # Apply granularity processing
            processed_grid = self._apply_granularity_processing(processed_grid, granularity, parameters)
            
            # Prepare color data
            color_data = self._prepare_color_data(processed_grid, color_by)
            
            # Render based on type
            if render_type == "3d":
                result = self._render_3d(processed_grid, color_data, **kwargs)
            elif render_type == "2d":
                result = self._render_2d(processed_grid, color_data, **kwargs)
            elif render_type == "cross_section":
                result = self._render_cross_section(processed_grid, color_data, **kwargs)
            else:
                raise ValueError(f"Unsupported render type: {render_type}")
            
            # Calculate render time
            render_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            render_result = RenderResult(
                success=True,
                render_type=render_type,
                data=result,
                metadata={
                    'color_by': color_by,
                    'voxel_size': processed_grid.voxel_size,
                    'dimensions': processed_grid.dimensions,
                    'render_engine': self.config.render_engine,
                    'level_of_detail': self.config.level_of_detail
                },
                render_time=render_time,
                voxel_count=processed_grid.solid_voxels
            )
            
            # Cache result if enabled
            if self.config.enable_caching:
                self._cache_render_result(render_result)
            
            logger.info(f"Voxel grid rendered successfully: {render_time:.2f}s, {processed_grid.solid_voxels} voxels")
            return render_result
            
        except Exception as e:
            logger.error(f"Error rendering voxel grid: {e}")
            return RenderResult(
                success=False,
                render_type=render_type,
                data=None,
                metadata={},
                render_time=0.0,
                voxel_count=0,
                error_message=str(e)
            )
    
    def render_fused_data(
        self,
        fused_data: Dict[Tuple[int, int, int], FusedVoxelData],
        voxel_grid: VoxelGrid,
        render_type: str = "3d",
        color_by: str = "quality",
        **kwargs
    ) -> RenderResult:
        """
        Render fused voxel data with enhanced visualization.
        
        Args:
            fused_data: Fused voxel data
            voxel_grid: Voxel grid structure
            render_type: Type of rendering
            color_by: Color mapping
            **kwargs: Additional parameters
            
        Returns:
            RenderResult: Rendering result
        """
        try:
            start_time = datetime.now()
            
            # Create enhanced voxel grid with fused data
            enhanced_grid = self._create_enhanced_grid(fused_data, voxel_grid)
            
            # Render enhanced grid
            result = self.render_voxel_grid(enhanced_grid, render_type, color_by, **kwargs)
            
            # Add fused data metadata
            result.metadata.update({
                'fused_voxels': len(fused_data),
                'defect_voxels': sum(1 for v in fused_data.values() if v.defect_count > 0),
                'data_completeness': np.mean([v.data_completeness for v in fused_data.values()]),
                'fusion_confidence': np.mean([v.fusion_confidence for v in fused_data.values()])
            })
            
            logger.info(f"Fused data rendered successfully: {len(fused_data)} fused voxels")
            return result
            
        except Exception as e:
            logger.error(f"Error rendering fused data: {e}")
            return RenderResult(
                success=False,
                render_type=render_type,
                data=None,
                metadata={},
                render_time=0.0,
                voxel_count=0,
                error_message=str(e)
            )
    
    def _validate_voxel_grid(self, voxel_grid: VoxelGrid) -> bool:
        """Validate voxel grid for rendering."""
        if not voxel_grid:
            return False
        
        if not hasattr(voxel_grid, 'voxels') or voxel_grid.voxels is None:
            return False
        
        if voxel_grid.solid_voxels == 0:
            return False
        
        return True
    
    def _apply_level_of_detail(self, voxel_grid: VoxelGrid) -> VoxelGrid:
        """Apply level of detail to reduce voxel count."""
        if self.config.level_of_detail == 1:
            return voxel_grid
        
        # Downsample voxel grid
        step = self.config.level_of_detail
        downsampled_voxels = voxel_grid.voxels[::step, ::step, ::step]
        
        # Create new voxel grid with downsampled data
        new_dimensions = downsampled_voxels.shape
        new_voxel_size = voxel_grid.voxel_size * step
        
        # Update process maps
        new_process_map = {}
        for key, data in voxel_grid.process_map.items():
            new_process_map[key] = data[::step, ::step, ::step]
        
        # Create new voxel grid
        new_grid = VoxelGrid(
            origin=voxel_grid.origin,
            dimensions=new_dimensions,
            voxel_size=new_voxel_size,
            bounds=voxel_grid.bounds,
            voxels=downsampled_voxels,
            material_map=voxel_grid.material_map[::step, ::step, ::step],
            process_map=new_process_map,
            creation_time=voxel_grid.creation_time,
            cad_file_path=voxel_grid.cad_file_path
        )
        
        # Recalculate statistics
        new_grid.total_voxels = np.prod(new_dimensions)
        new_grid.solid_voxels = np.sum(downsampled_voxels > 0)
        new_grid.void_voxels = new_grid.total_voxels - new_grid.solid_voxels
        
        return new_grid
    
    def _prepare_color_data(self, voxel_grid: VoxelGrid, color_by: str) -> np.ndarray:
        """Prepare color data for rendering."""
        if color_by == "quality":
            return voxel_grid.process_map.get('quality_score', np.ones_like(voxel_grid.voxels) * 100)
        elif color_by == "defect":
            # Create defect probability map
            defect_map = np.zeros_like(voxel_grid.voxels)
            # This would be populated from fused data in real implementation
            return defect_map
        elif color_by == "process":
            return voxel_grid.process_map.get('laser_power', np.ones_like(voxel_grid.voxels) * 300)
        elif color_by == "material":
            return voxel_grid.material_map
        else:
            return voxel_grid.voxels
    
    def _render_3d(self, voxel_grid: VoxelGrid, color_data: np.ndarray, **kwargs) -> Any:
        """Render 3D voxel visualization."""
        if self.config.render_engine == "pyvista":
            return self._render_3d_pyvista(voxel_grid, color_data, **kwargs)
        elif self.config.render_engine == "plotly":
            return self._render_3d_plotly(voxel_grid, color_data, **kwargs)
        elif self.config.render_engine == "matplotlib":
            return self._render_3d_matplotlib(voxel_grid, color_data, **kwargs)
        else:
            raise ValueError(f"Unsupported render engine: {self.config.render_engine}")
    
    def _render_3d_plotly(self, voxel_grid: VoxelGrid, color_data: np.ndarray, **kwargs) -> go.Figure:
        """Render 3D visualization using Plotly."""
        # Get solid voxel positions
        solid_voxels = np.where(voxel_grid.voxels > 0)
        
        if len(solid_voxels[0]) == 0:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(title="No voxels to display")
            return fig
        
        # Convert to world coordinates
        x_coords = voxel_grid.origin[0] + solid_voxels[0] * voxel_grid.voxel_size
        y_coords = voxel_grid.origin[1] + solid_voxels[1] * voxel_grid.voxel_size
        z_coords = voxel_grid.origin[2] + solid_voxels[2] * voxel_grid.voxel_size
        
        # Get color values
        color_values = color_data[solid_voxels]
        
        # Create 3D scatter plot
        fig = go.Figure(data=go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=voxel_grid.voxel_size * 2,  # Scale marker size
                color=color_values,
                colorscale=self.color_maps.get('default', 'viridis'),
                opacity=self.config.opacity,
                colorbar=dict(title="Value")
            ),
            text=[f"Voxel ({x:.1f}, {y:.1f}, {z:.1f})<br>Value: {c:.2f}" 
                  for x, y, z, c in zip(x_coords, y_coords, z_coords, color_values)],
            hovertemplate="%{text}<extra></extra>"
        ))
        
        # Update layout
        fig.update_layout(
            title="3D Voxel Visualization",
            scene=dict(
                xaxis_title="X (mm)",
                yaxis_title="Y (mm)",
                zaxis_title="Z (mm)",
                aspectmode="data"
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def _render_3d_matplotlib(self, voxel_grid: VoxelGrid, color_data: np.ndarray, **kwargs) -> plt.Figure:
        """Render 3D visualization using Matplotlib."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get solid voxel positions
        solid_voxels = np.where(voxel_grid.voxels > 0)
        
        if len(solid_voxels[0]) == 0:
            ax.text(0.5, 0.5, 0.5, "No voxels to display", transform=ax.transAxes)
            return fig
        
        # Convert to world coordinates
        x_coords = voxel_grid.origin[0] + solid_voxels[0] * voxel_grid.voxel_size
        y_coords = voxel_grid.origin[1] + solid_voxels[1] * voxel_grid.voxel_size
        z_coords = voxel_grid.origin[2] + solid_voxels[2] * voxel_grid.voxel_size
        
        # Get color values
        color_values = color_data[solid_voxels]
        
        # Create 3D scatter plot
        scatter = ax.scatter(
            x_coords, y_coords, z_coords,
            c=color_values,
            cmap=self.color_maps.get('default', 'viridis'),
            alpha=self.config.opacity,
            s=voxel_grid.voxel_size * 10  # Scale marker size
        )
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label="Value")
        
        # Set labels and title
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.set_title("3D Voxel Visualization")
        
        return fig
    
    def _render_3d_pyvista(self, voxel_grid: VoxelGrid, color_data: np.ndarray, **kwargs) -> pv.Plotter:
        """Render 3D visualization using PyVista's voxel model for superior performance."""
        # Create PyVista plotter with off_screen rendering for screenshots
        plotter = pv.Plotter(off_screen=True)
        
        # Check if we have a PyVista voxel model (preferred)
        if hasattr(voxel_grid, 'pyvista_voxel_model') and voxel_grid.pyvista_voxel_model is not None:
            logger.info("Using PyVista voxel model for rendering")
            
            # Use the PyVista voxel model directly (like legos!)
            voxel_model = voxel_grid.pyvista_voxel_model
            
            # Add color data if available
            if color_data is not None and len(color_data.shape) == 3:
                # Map color data to voxel model
                # This is more complex - we'd need to map our grid colors to PyVista cells
                # For now, use a simple approach
                plotter.add_mesh(voxel_model, 
                               color='lightblue',
                               opacity=self.config.opacity,
                               show_edges=False)
            else:
                # Add voxel model without color mapping
                plotter.add_mesh(voxel_model, 
                               color='lightblue',
                               opacity=self.config.opacity,
                               show_edges=False)
        
        else:
            # Fallback to point cloud rendering
            logger.info("Using point cloud fallback for rendering")
            
            # Get solid voxel positions
            solid_voxels = np.where(voxel_grid.voxels > 0)
            
            if len(solid_voxels[0]) == 0:
                plotter.add_text("No voxels to display", position='upper_left')
                return plotter
            
            # Convert voxel coordinates to world coordinates
            world_coords = np.array([
                voxel_grid.origin[0] + solid_voxels[0] * voxel_grid.voxel_size,
                voxel_grid.origin[1] + solid_voxels[1] * voxel_grid.voxel_size,
                voxel_grid.origin[2] + solid_voxels[2] * voxel_grid.voxel_size
            ]).T
            
            # Create point cloud
            points = pv.PolyData(world_coords)
            
            # Add color data if available
            if color_data is not None and len(color_data.shape) == 3:
                # Extract color values for solid voxels
                color_values = color_data[solid_voxels]
                points['color_data'] = color_values
                
                # Add voxel grid with color mapping
                plotter.add_mesh(points, scalars='color_data', 
                               point_size=voxel_grid.voxel_size * 10,
                               render_points_as_spheres=True,
                               cmap=self.config.color_scheme,
                               opacity=self.config.opacity)
            else:
                # Add voxel grid without color mapping
                plotter.add_mesh(points, 
                               point_size=voxel_grid.voxel_size * 10,
                               render_points_as_spheres=True,
                               color='lightblue',
                               opacity=self.config.opacity)
        
        # Add axes
        if self.config.show_axes:
            plotter.add_axes()
        
        # Add grid
        if self.config.show_grid:
            plotter.show_grid()
        
        # Set camera position for better view
        plotter.camera_position = 'iso'  # Use 'iso' instead of 'isometric'
        
        # Add title
        plotter.add_text("3D Voxel Visualization (PyVista)", position='upper_left')
        
        return plotter
    
    def _render_2d(self, voxel_grid: VoxelGrid, color_data: np.ndarray, **kwargs) -> Any:
        """Render 2D slice visualization."""
        slice_axis = kwargs.get('slice_axis', 'z')  # 'x', 'y', or 'z'
        slice_index = kwargs.get('slice_index', voxel_grid.dimensions[2] // 2)
        
        if self.config.render_engine == "plotly":
            return self._render_2d_plotly(voxel_grid, color_data, slice_axis, slice_index)
        elif self.config.render_engine == "matplotlib":
            return self._render_2d_matplotlib(voxel_grid, color_data, slice_axis, slice_index)
        else:
            raise ValueError(f"Unsupported render engine: {self.config.render_engine}")
    
    def _render_2d_plotly(self, voxel_grid: VoxelGrid, color_data: np.ndarray, slice_axis: str, slice_index: int) -> go.Figure:
        """Render 2D slice using Plotly."""
        # Extract slice
        if slice_axis == 'x':
            slice_data = color_data[slice_index, :, :]
            x_title, y_title = "Y (mm)", "Z (mm)"
        elif slice_axis == 'y':
            slice_data = color_data[:, slice_index, :]
            x_title, y_title = "X (mm)", "Z (mm)"
        else:  # z
            slice_data = color_data[:, :, slice_index]
            x_title, y_title = "X (mm)", "Y (mm)"
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=slice_data,
            colorscale=self.color_maps.get('default', 'viridis'),
            showscale=True
        ))
        
        fig.update_layout(
            title=f"2D Slice Visualization ({slice_axis.upper()}={slice_index})",
            xaxis_title=x_title,
            yaxis_title=y_title
        )
        
        return fig
    
    def _render_2d_matplotlib(self, voxel_grid: VoxelGrid, color_data: np.ndarray, slice_axis: str, slice_index: int) -> plt.Figure:
        """Render 2D slice using Matplotlib."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract slice
        if slice_axis == 'x':
            slice_data = color_data[slice_index, :, :]
            x_title, y_title = "Y (mm)", "Z (mm)"
        elif slice_axis == 'y':
            slice_data = color_data[:, slice_index, :]
            x_title, y_title = "X (mm)", "Z (mm)"
        else:  # z
            slice_data = color_data[:, :, slice_index]
            x_title, y_title = "X (mm)", "Y (mm)"
        
        # Create heatmap
        im = ax.imshow(slice_data, cmap=self.color_maps.get('default', 'viridis'))
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label="Value")
        
        # Set labels and title
        ax.set_xlabel(x_title)
        ax.set_ylabel(y_title)
        ax.set_title(f"2D Slice Visualization ({slice_axis.upper()}={slice_index})")
        
        return fig
    
    def _render_cross_section(self, voxel_grid: VoxelGrid, color_data: np.ndarray, **kwargs) -> Any:
        """Render cross-section visualization."""
        # For now, use 2D rendering
        return self._render_2d(voxel_grid, color_data, **kwargs)
    
    def _create_enhanced_grid(
        self, 
        fused_data: Dict[Tuple[int, int, int], FusedVoxelData], 
        voxel_grid: VoxelGrid
    ) -> VoxelGrid:
        """Create enhanced voxel grid with fused data."""
        # Create enhanced process map with fused data
        enhanced_process_map = voxel_grid.process_map.copy()
        
        # Add fused data to process map
        for voxel_idx, voxel_data in fused_data.items():
            if voxel_data.overall_quality_score is not None:
                enhanced_process_map['quality_score'][voxel_idx] = voxel_data.overall_quality_score
            
            if voxel_data.ct_defect_probability is not None:
                enhanced_process_map['defect_probability'][voxel_idx] = voxel_data.ct_defect_probability
            
            if voxel_data.ispm_temperature is not None:
                enhanced_process_map['temperature'][voxel_idx] = voxel_data.ispm_temperature
        
        # Create enhanced voxel grid
        enhanced_grid = VoxelGrid(
            origin=voxel_grid.origin,
            dimensions=voxel_grid.dimensions,
            voxel_size=voxel_grid.voxel_size,
            bounds=voxel_grid.bounds,
            voxels=voxel_grid.voxels,
            material_map=voxel_grid.material_map,
            process_map=enhanced_process_map,
            creation_time=voxel_grid.creation_time,
            cad_file_path=voxel_grid.cad_file_path
        )
        
        # Copy statistics
        enhanced_grid.total_voxels = voxel_grid.total_voxels
        enhanced_grid.solid_voxels = voxel_grid.solid_voxels
        enhanced_grid.void_voxels = voxel_grid.void_voxels
        
        return enhanced_grid
    
    def _cache_render_result(self, render_result: RenderResult):
        """Cache render result for performance."""
        cache_key = f"{render_result.render_type}_{render_result.metadata.get('color_by', 'default')}"
        self.render_cache[cache_key] = render_result
    
    def export_render_result(self, render_result: RenderResult, output_path: str):
        """Export render result to file."""
        try:
            if not render_result.success:
                raise ValueError("Cannot export failed render result")
            
            if self.config.render_engine == "plotly":
                if render_result.data:
                    render_result.data.write_html(output_path)
            elif self.config.render_engine == "matplotlib":
                if render_result.data:
                    render_result.data.savefig(output_path, dpi=self.config.image_dpi, 
                                             format=self.config.image_format, bbox_inches='tight')
            
            logger.info(f"Render result exported to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting render result: {e}")
            raise
    
    def get_render_statistics(self) -> Dict[str, Any]:
        """Get rendering statistics."""
        return {
            'cache_size': len(self.render_cache),
            'render_engine': self.config.render_engine,
            'max_voxels_display': self.config.max_voxels_display,
            'level_of_detail': self.config.level_of_detail,
            'color_maps': self.color_maps
        }
    
    def _apply_granularity_processing(
        self, 
        voxel_grid: VoxelGrid, 
        granularity: str, 
        parameters: List[str] = None
    ) -> VoxelGrid:
        """
        Apply granularity processing to voxel grid based on user selection.
        
        Args:
            voxel_grid: Input voxel grid
            granularity: Granularity level (per_hatch, per_layer, per_build, custom)
            parameters: List of parameters to include
            
        Returns:
            Processed voxel grid with appropriate granularity
        """
        try:
            if granularity == "per_hatch":
                return self._process_per_hatch_granularity(voxel_grid, parameters)
            elif granularity == "per_layer":
                return self._process_per_layer_granularity(voxel_grid, parameters)
            elif granularity == "per_build":
                return self._process_per_build_granularity(voxel_grid, parameters)
            elif granularity == "custom":
                return self._process_custom_granularity(voxel_grid, parameters)
            else:
                logger.warning(f"Unknown granularity: {granularity}, using per_hatch")
                return self._process_per_hatch_granularity(voxel_grid, parameters)
                
        except Exception as e:
            logger.error(f"Error applying granularity processing: {e}")
            return voxel_grid
    
    def _process_per_hatch_granularity(self, voxel_grid: VoxelGrid, parameters: List[str] = None) -> VoxelGrid:
        """Process voxel grid with per-hatch granularity (most detailed)."""
        # Per-hatch granularity uses the raw extracted data as-is
        # This is the most detailed level with exact process parameters
        logger.info("Processing voxel grid with per-hatch granularity")
        return voxel_grid
    
    def _process_per_layer_granularity(self, voxel_grid: VoxelGrid, parameters: List[str] = None) -> VoxelGrid:
        """Process voxel grid with per-layer granularity (averaged by layer)."""
        logger.info("Processing voxel grid with per-layer granularity")
        
        # Group voxels by layer and average process parameters
        if hasattr(voxel_grid, 'process_map') and voxel_grid.process_map:
            layer_averaged_map = {}
            
            for param_name, param_data in voxel_grid.process_map.items():
                if parameters and param_name not in parameters:
                    continue
                    
                # Average parameters by layer
                layer_averaged_map[param_name] = self._average_parameters_by_layer(param_data, voxel_grid)
            
            # Update the process map with layer-averaged data
            voxel_grid.process_map = layer_averaged_map
        
        return voxel_grid
    
    def _process_per_build_granularity(self, voxel_grid: VoxelGrid, parameters: List[str] = None) -> VoxelGrid:
        """Process voxel grid with per-build granularity (global averages)."""
        logger.info("Processing voxel grid with per-build granularity")
        
        # Calculate global averages for all process parameters
        if hasattr(voxel_grid, 'process_map') and voxel_grid.process_map:
            global_averaged_map = {}
            
            for param_name, param_data in voxel_grid.process_map.items():
                if parameters and param_name not in parameters:
                    continue
                    
                # Calculate global average
                global_avg = np.mean(param_data) if param_data.size > 0 else 0.0
                global_averaged_map[param_name] = np.full_like(param_data, global_avg)
            
            # Update the process map with global-averaged data
            voxel_grid.process_map = global_averaged_map
        
        return voxel_grid
    
    def _process_custom_granularity(self, voxel_grid: VoxelGrid, parameters: List[str] = None) -> VoxelGrid:
        """Process voxel grid with custom granularity (user-defined regions)."""
        logger.info("Processing voxel grid with custom granularity")
        
        # Custom granularity allows user to define spatial regions
        # For now, implement as per-layer but this can be extended
        # to support user-defined spatial regions
        return self._process_per_layer_granularity(voxel_grid, parameters)
    
    def _average_parameters_by_layer(self, param_data: np.ndarray, voxel_grid: VoxelGrid) -> np.ndarray:
        """Average process parameters by layer."""
        try:
            # Assuming Z-axis represents layers
            if len(param_data.shape) >= 3:
                # Average along Z-axis (layers)
                layer_averaged = np.mean(param_data, axis=2, keepdims=True)
                # Broadcast back to original shape
                return np.broadcast_to(layer_averaged, param_data.shape)
            else:
                return param_data
        except Exception as e:
            logger.warning(f"Error averaging parameters by layer: {e}")
            return param_data
