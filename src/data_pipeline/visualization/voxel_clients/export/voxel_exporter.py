"""
Voxel Data Export and Sharing for PBF-LB/M Visualization

This module provides comprehensive voxel data export and sharing capabilities
for PBF-LB/M research. It supports multiple export formats and sharing methods.
"""

import numpy as np
import pandas as pd
import json
import pickle
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
import zipfile
import tarfile
import base64
import io

from src.core.domain.value_objects.voxel_coordinates import VoxelCoordinates
from ..core.cad_voxelizer import VoxelGrid
from ..core.multi_modal_fusion import FusedVoxelData
from ..analysis.spatial_quality_analyzer import SpatialQualityMetrics
from ..analysis.defect_detector_3d import DefectDetectionResult
from ..analysis.porosity_analyzer import PorosityAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for voxel data export."""
    
    # Export formats
    supported_formats: List[str] = None
    default_format: str = "npz"
    compression_level: int = 6
    
    # Data processing
    include_metadata: bool = True
    include_statistics: bool = True
    include_visualization_data: bool = True
    normalize_coordinates: bool = True
    
    # Performance settings
    chunk_size: int = 10000
    max_file_size_mb: float = 100.0
    enable_compression: bool = True
    
    # Sharing settings
    enable_sharing: bool = True
    sharing_platforms: List[str] = None
    anonymize_data: bool = False


@dataclass
class ExportResult:
    """Result of export operation."""
    
    success: bool
    export_format: str
    file_path: str
    file_size: int
    export_time: float
    data_summary: Dict[str, Any]
    error_message: Optional[str] = None


class VoxelExporter:
    """
    Voxel data exporter and sharing system for PBF-LB/M research.
    
    This class provides comprehensive export capabilities including:
    - Multiple export formats (NPZ, HDF5, JSON, CSV, etc.)
    - Data compression and optimization
    - Metadata and statistics inclusion
    - Visualization data export
    - Sharing and collaboration features
    - Batch export operations
    """
    
    def __init__(self, config: ExportConfig = None):
        """Initialize the voxel exporter."""
        self.config = config or ExportConfig()
        if self.config.supported_formats is None:
            self.config.supported_formats = [
                'npz', 'h5', 'hdf5', 'json', 'csv', 'txt', 'zip', 'tar.gz'
            ]
        if self.config.sharing_platforms is None:
            self.config.sharing_platforms = ['local', 'cloud', 'email']
        
        self.export_cache = {}
        self.export_history = []
        
        logger.info("Voxel Exporter initialized")
    
    def export_voxel_grid(
        self, 
        voxel_grid: VoxelGrid, 
        output_path: str,
        export_format: str = None
    ) -> ExportResult:
        """
        Export voxel grid to file.
        
        Args:
            voxel_grid: Voxel grid to export
            output_path: Output file path
            export_format: Export format (auto-detected if None)
            
        Returns:
            ExportResult: Export operation result
        """
        try:
            start_time = datetime.now()
            
            # Determine export format
            if export_format is None:
                export_format = self._detect_format_from_path(output_path)
            
            if export_format not in self.config.supported_formats:
                raise ValueError(f"Unsupported export format: {export_format}")
            
            # Prepare export data
            export_data = self._prepare_voxel_grid_data(voxel_grid)
            
            # Export based on format
            if export_format == 'npz':
                self._export_npz(export_data, output_path)
            elif export_format in ['h5', 'hdf5']:
                self._export_hdf5(export_data, output_path)
            elif export_format == 'json':
                self._export_json(export_data, output_path)
            elif export_format == 'csv':
                self._export_csv(export_data, output_path)
            elif export_format == 'txt':
                self._export_text(export_data, output_path)
            elif export_format == 'zip':
                self._export_zip(export_data, output_path)
            elif export_format == 'tar.gz':
                self._export_tar_gz(export_data, output_path)
            else:
                raise ValueError(f"Export format not implemented: {export_format}")
            
            # Calculate export time and file size
            export_time = (datetime.now() - start_time).total_seconds()
            file_size = Path(output_path).stat().st_size
            
            # Create data summary
            data_summary = {
                'voxel_grid_dimensions': voxel_grid.dimensions,
                'voxel_size': voxel_grid.voxel_size,
                'total_voxels': voxel_grid.total_voxels,
                'solid_voxels': voxel_grid.solid_voxels,
                'void_voxels': voxel_grid.void_voxels,
                'fill_ratio': voxel_grid.solid_voxels / voxel_grid.total_voxels,
                'export_format': export_format,
                'compression_enabled': self.config.enable_compression
            }
            
            # Create result
            result = ExportResult(
                success=True,
                export_format=export_format,
                file_path=output_path,
                file_size=file_size,
                export_time=export_time,
                data_summary=data_summary
            )
            
            # Store in history
            self.export_history.append(result)
            
            logger.info(f"Voxel grid exported successfully: {output_path} ({export_time:.2f}s, {file_size} bytes)")
            return result
            
        except Exception as e:
            logger.error(f"Error exporting voxel grid: {e}")
            return ExportResult(
                success=False,
                export_format=export_format or "unknown",
                file_path=output_path,
                file_size=0,
                export_time=0.0,
                data_summary={},
                error_message=str(e)
            )
    
    def export_fused_data(
        self, 
        fused_data: Dict[Tuple[int, int, int], FusedVoxelData], 
        output_path: str,
        export_format: str = None
    ) -> ExportResult:
        """Export fused voxel data to file."""
        try:
            start_time = datetime.now()
            
            # Determine export format
            if export_format is None:
                export_format = self._detect_format_from_path(output_path)
            
            # Prepare export data
            export_data = self._prepare_fused_data(fused_data)
            
            # Export based on format
            if export_format == 'npz':
                self._export_npz(export_data, output_path)
            elif export_format in ['h5', 'hdf5']:
                self._export_hdf5(export_data, output_path)
            elif export_format == 'json':
                self._export_json(export_data, output_path)
            elif export_format == 'csv':
                self._export_csv(export_data, output_path)
            else:
                raise ValueError(f"Export format not supported for fused data: {export_format}")
            
            # Calculate export time and file size
            export_time = (datetime.now() - start_time).total_seconds()
            file_size = Path(output_path).stat().st_size
            
            # Create data summary
            data_summary = {
                'fused_voxels': len(fused_data),
                'defect_voxels': sum(1 for v in fused_data.values() if v.defect_count > 0),
                'export_format': export_format,
                'compression_enabled': self.config.enable_compression
            }
            
            # Create result
            result = ExportResult(
                success=True,
                export_format=export_format,
                file_path=output_path,
                file_size=file_size,
                export_time=export_time,
                data_summary=data_summary
            )
            
            # Store in history
            self.export_history.append(result)
            
            logger.info(f"Fused data exported successfully: {output_path} ({export_time:.2f}s, {file_size} bytes)")
            return result
            
        except Exception as e:
            logger.error(f"Error exporting fused data: {e}")
            return ExportResult(
                success=False,
                export_format=export_format or "unknown",
                file_path=output_path,
                file_size=0,
                export_time=0.0,
                data_summary={},
                error_message=str(e)
            )
    
    def export_analysis_results(
        self, 
        analysis_results: Dict[str, Any], 
        output_path: str,
        export_format: str = None
    ) -> ExportResult:
        """Export analysis results to file."""
        try:
            start_time = datetime.now()
            
            # Determine export format
            if export_format is None:
                export_format = self._detect_format_from_path(output_path)
            
            # Prepare export data
            export_data = self._prepare_analysis_results(analysis_results)
            
            # Export based on format
            if export_format == 'json':
                self._export_json(export_data, output_path)
            elif export_format == 'csv':
                self._export_csv(export_data, output_path)
            elif export_format == 'txt':
                self._export_text(export_data, output_path)
            else:
                raise ValueError(f"Export format not supported for analysis results: {export_format}")
            
            # Calculate export time and file size
            export_time = (datetime.now() - start_time).total_seconds()
            file_size = Path(output_path).stat().st_size
            
            # Create data summary
            data_summary = {
                'analysis_types': list(analysis_results.keys()),
                'export_format': export_format,
                'compression_enabled': self.config.enable_compression
            }
            
            # Create result
            result = ExportResult(
                success=True,
                export_format=export_format,
                file_path=output_path,
                file_size=file_size,
                export_time=export_time,
                data_summary=data_summary
            )
            
            # Store in history
            self.export_history.append(result)
            
            logger.info(f"Analysis results exported successfully: {output_path} ({export_time:.2f}s, {file_size} bytes)")
            return result
            
        except Exception as e:
            logger.error(f"Error exporting analysis results: {e}")
            return ExportResult(
                success=False,
                export_format=export_format or "unknown",
                file_path=output_path,
                file_size=0,
                export_time=0.0,
                data_summary={},
                error_message=str(e)
            )
    
    def export_visualization_data(
        self, 
        visualization_data: Any, 
        output_path: str,
        export_format: str = None
    ) -> ExportResult:
        """Export visualization data to file."""
        try:
            start_time = datetime.now()
            
            # Determine export format
            if export_format is None:
                export_format = self._detect_format_from_path(output_path)
            
            # Prepare export data
            export_data = self._prepare_visualization_data(visualization_data)
            
            # Export based on format
            if export_format == 'json':
                self._export_json(export_data, output_path)
            elif export_format == 'html':
                self._export_html(export_data, output_path)
            elif export_format == 'png':
                self._export_image(export_data, output_path, 'png')
            elif export_format == 'jpg':
                self._export_image(export_data, output_path, 'jpg')
            elif export_format == 'svg':
                self._export_image(export_data, output_path, 'svg')
            else:
                raise ValueError(f"Export format not supported for visualization: {export_format}")
            
            # Calculate export time and file size
            export_time = (datetime.now() - start_time).total_seconds()
            file_size = Path(output_path).stat().st_size
            
            # Create data summary
            data_summary = {
                'visualization_type': type(visualization_data).__name__,
                'export_format': export_format,
                'compression_enabled': self.config.enable_compression
            }
            
            # Create result
            result = ExportResult(
                success=True,
                export_format=export_format,
                file_path=output_path,
                file_size=file_size,
                export_time=export_time,
                data_summary=data_summary
            )
            
            # Store in history
            self.export_history.append(result)
            
            logger.info(f"Visualization data exported successfully: {output_path} ({export_time:.2f}s, {file_size} bytes)")
            return result
            
        except Exception as e:
            logger.error(f"Error exporting visualization data: {e}")
            return ExportResult(
                success=False,
                export_format=export_format or "unknown",
                file_path=output_path,
                file_size=0,
                export_time=0.0,
                data_summary={},
                error_message=str(e)
            )
    
    def batch_export(
        self, 
        data_items: List[Dict[str, Any]], 
        output_directory: str,
        export_format: str = None
    ) -> List[ExportResult]:
        """Export multiple data items in batch."""
        results = []
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, data_item in enumerate(data_items):
            try:
                data_type = data_item.get('type', 'unknown')
                data = data_item.get('data')
                filename = data_item.get('filename', f"{data_type}_{i}")
                
                # Determine file extension
                if export_format:
                    ext = export_format
                else:
                    ext = self.config.default_format
                
                output_path = output_dir / f"{filename}.{ext}"
                
                # Export based on data type
                if data_type == 'voxel_grid':
                    result = self.export_voxel_grid(data, str(output_path), ext)
                elif data_type == 'fused_data':
                    result = self.export_fused_data(data, str(output_path), ext)
                elif data_type == 'analysis_results':
                    result = self.export_analysis_results(data, str(output_path), ext)
                elif data_type == 'visualization_data':
                    result = self.export_visualization_data(data, str(output_path), ext)
                else:
                    result = ExportResult(
                        success=False,
                        export_format=ext,
                        file_path=str(output_path),
                        file_size=0,
                        export_time=0.0,
                        data_summary={},
                        error_message=f"Unknown data type: {data_type}"
                    )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error in batch export item {i}: {e}")
                results.append(ExportResult(
                    success=False,
                    export_format=ext,
                    file_path=str(output_path),
                    file_size=0,
                    export_time=0.0,
                    data_summary={},
                    error_message=str(e)
                ))
        
        logger.info(f"Batch export completed: {len(results)} items processed")
        return results
    
    def share_data(
        self, 
        data: Any, 
        sharing_platform: str = "local",
        **kwargs
    ) -> Dict[str, Any]:
        """Share data through various platforms."""
        try:
            if sharing_platform not in self.config.sharing_platforms:
                raise ValueError(f"Unsupported sharing platform: {sharing_platform}")
            
            if sharing_platform == "local":
                return self._share_local(data, **kwargs)
            elif sharing_platform == "cloud":
                return self._share_cloud(data, **kwargs)
            elif sharing_platform == "email":
                return self._share_email(data, **kwargs)
            else:
                raise ValueError(f"Sharing platform not implemented: {sharing_platform}")
                
        except Exception as e:
            logger.error(f"Error sharing data: {e}")
            return {'success': False, 'error': str(e)}
    
    # Private methods
    def _detect_format_from_path(self, file_path: str) -> str:
        """Detect export format from file path extension."""
        ext = Path(file_path).suffix.lower().lstrip('.')
        if ext in self.config.supported_formats:
            return ext
        else:
            return self.config.default_format
    
    def _prepare_voxel_grid_data(self, voxel_grid: VoxelGrid) -> Dict[str, Any]:
        """Prepare voxel grid data for export."""
        data = {
            'voxels': voxel_grid.voxels,
            'material_map': voxel_grid.material_map,
            'process_map': voxel_grid.process_map,
            'origin': voxel_grid.origin,
            'dimensions': voxel_grid.dimensions,
            'voxel_size': voxel_grid.voxel_size,
            'bounds': voxel_grid.bounds,
            'creation_time': voxel_grid.creation_time.isoformat(),
            'cad_file_path': voxel_grid.cad_file_path
        }
        
        if self.config.include_metadata:
            data['metadata'] = {
                'total_voxels': voxel_grid.total_voxels,
                'solid_voxels': voxel_grid.solid_voxels,
                'void_voxels': voxel_grid.void_voxels,
                'fill_ratio': voxel_grid.solid_voxels / voxel_grid.total_voxels,
                'export_timestamp': datetime.now().isoformat(),
                'exporter_version': '1.0.0'
            }
        
        return data
    
    def _prepare_fused_data(self, fused_data: Dict[Tuple[int, int, int], FusedVoxelData]) -> Dict[str, Any]:
        """Prepare fused data for export."""
        data = {}
        
        for voxel_idx, voxel_data in fused_data.items():
            # Convert tuple key to string for JSON serialization
            key = f"{voxel_idx[0]}_{voxel_idx[1]}_{voxel_idx[2]}"
            data[key] = asdict(voxel_data)
            
            # Convert datetime objects to strings
            if 'build_time' in data[key]:
                data[key]['build_time'] = data[key]['build_time'].isoformat()
            if 'last_updated' in data[key]:
                data[key]['last_updated'] = data[key]['last_updated'].isoformat()
        
        if self.config.include_metadata:
            data['_metadata'] = {
                'total_voxels': len(fused_data),
                'defect_voxels': sum(1 for v in fused_data.values() if v.defect_count > 0),
                'export_timestamp': datetime.now().isoformat(),
                'exporter_version': '1.0.0'
            }
        
        return data
    
    def _prepare_analysis_results(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare analysis results for export."""
        data = analysis_results.copy()
        
        if self.config.include_metadata:
            data['_metadata'] = {
                'analysis_types': list(analysis_results.keys()),
                'export_timestamp': datetime.now().isoformat(),
                'exporter_version': '1.0.0'
            }
        
        return data
    
    def _prepare_visualization_data(self, visualization_data: Any) -> Dict[str, Any]:
        """Prepare visualization data for export."""
        # Convert visualization data to serializable format
        if hasattr(visualization_data, 'to_dict'):
            data = visualization_data.to_dict()
        elif hasattr(visualization_data, 'to_json'):
            data = json.loads(visualization_data.to_json())
        else:
            data = {'visualization_data': str(visualization_data)}
        
        if self.config.include_metadata:
            data['_metadata'] = {
                'visualization_type': type(visualization_data).__name__,
                'export_timestamp': datetime.now().isoformat(),
                'exporter_version': '1.0.0'
            }
        
        return data
    
    def _export_npz(self, data: Dict[str, Any], output_path: str):
        """Export data to NumPy compressed format."""
        # Convert data to numpy arrays where possible
        np_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                np_data[key] = value
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                try:
                    np_data[key] = np.array(value)
                except:
                    np_data[key] = value
            else:
                np_data[key] = value
        
        np.savez_compressed(output_path, **np_data)
    
    def _export_hdf5(self, data: Dict[str, Any], output_path: str):
        """Export data to HDF5 format."""
        with h5py.File(output_path, 'w') as f:
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value, compression='gzip' if self.config.enable_compression else None)
                elif isinstance(value, (list, tuple)):
                    try:
                        f.create_dataset(key, data=np.array(value), compression='gzip' if self.config.enable_compression else None)
                    except:
                        f.attrs[key] = str(value)
                else:
                    f.attrs[key] = str(value)
    
    def _export_json(self, data: Dict[str, Any], output_path: str):
        """Export data to JSON format."""
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _export_csv(self, data: Dict[str, Any], output_path: str):
        """Export data to CSV format."""
        # Convert data to DataFrame
        df_data = []
        for key, value in data.items():
            if key.startswith('_'):
                continue
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    df_data.append({'key': f"{key}.{sub_key}", 'value': sub_value})
            else:
                df_data.append({'key': key, 'value': value})
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_path, index=False)
    
    def _export_text(self, data: Dict[str, Any], output_path: str):
        """Export data to text format."""
        with open(output_path, 'w') as f:
            f.write("=== VOXEL DATA EXPORT ===\n\n")
            f.write(f"Export Timestamp: {datetime.now().isoformat()}\n\n")
            
            for key, value in data.items():
                if key.startswith('_'):
                    continue
                f.write(f"{key}:\n")
                f.write(f"{value}\n\n")
    
    def _export_zip(self, data: Dict[str, Any], output_path: str):
        """Export data to ZIP format."""
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for key, value in data.items():
                if isinstance(value, str):
                    zip_file.writestr(f"{key}.txt", value)
                else:
                    zip_file.writestr(f"{key}.json", json.dumps(value, default=str))
    
    def _export_tar_gz(self, data: Dict[str, Any], output_path: str):
        """Export data to TAR.GZ format."""
        with tarfile.open(output_path, 'w:gz') as tar_file:
            for key, value in data.items():
                if isinstance(value, str):
                    info = tarfile.TarInfo(name=f"{key}.txt")
                    info.size = len(value.encode())
                    tar_file.addfile(info, io.BytesIO(value.encode()))
                else:
                    json_str = json.dumps(value, default=str)
                    info = tarfile.TarInfo(name=f"{key}.json")
                    info.size = len(json_str.encode())
                    tar_file.addfile(info, io.BytesIO(json_str.encode()))
    
    def _export_html(self, data: Dict[str, Any], output_path: str):
        """Export data to HTML format."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Voxel Data Export</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
                .data-section {{ margin: 20px 0; }}
                .data-item {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Voxel Data Export</h1>
                <p>Export Timestamp: {datetime.now().isoformat()}</p>
            </div>
        """
        
        for key, value in data.items():
            if key.startswith('_'):
                continue
            html_content += f"""
            <div class="data-section">
                <h2>{key}</h2>
                <div class="data-item">
                    <pre>{json.dumps(value, indent=2, default=str)}</pre>
                </div>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _export_image(self, data: Dict[str, Any], output_path: str, format: str):
        """Export data as image (placeholder implementation)."""
        # This would need to be implemented based on the visualization library used
        logger.warning(f"Image export not implemented for format: {format}")
    
    def _share_local(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Share data locally (placeholder implementation)."""
        return {
            'success': True,
            'platform': 'local',
            'message': 'Data shared locally',
            'timestamp': datetime.now().isoformat()
        }
    
    def _share_cloud(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Share data via cloud platform (placeholder implementation)."""
        return {
            'success': True,
            'platform': 'cloud',
            'message': 'Data shared via cloud',
            'timestamp': datetime.now().isoformat()
        }
    
    def _share_email(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Share data via email (placeholder implementation)."""
        return {
            'success': True,
            'platform': 'email',
            'message': 'Data shared via email',
            'timestamp': datetime.now().isoformat()
        }
    
    def get_export_statistics(self) -> Dict[str, Any]:
        """Get export statistics."""
        return {
            'total_exports': len(self.export_history),
            'successful_exports': sum(1 for r in self.export_history if r.success),
            'failed_exports': sum(1 for r in self.export_history if not r.success),
            'total_export_time': sum(r.export_time for r in self.export_history),
            'total_export_size': sum(r.file_size for r in self.export_history),
            'supported_formats': self.config.supported_formats,
            'sharing_platforms': self.config.sharing_platforms
        }
