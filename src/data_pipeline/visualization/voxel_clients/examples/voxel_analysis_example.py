"""
Voxel Analysis Example for PBF-LB/M Research

This example demonstrates how to use the voxel visualization and analysis system
for comprehensive PBF-LB/M research. It shows the complete workflow from CAD
voxelization to spatially-resolved quality analysis.
"""

import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Import voxel visualization components
from ..core.cad_voxelizer import CADVoxelizer, VoxelizationConfig
from ..core.multi_modal_fusion import MultiModalFusion
from ..core.voxel_process_controller import VoxelProcessController, ProcessControlConfig, ControlMode, OptimizationObjective
from ..analysis.spatial_quality_analyzer import SpatialQualityAnalyzer, QualityAnalysisConfig

# Import domain objects
from src.core.domain.value_objects.process_parameters import ProcessParameters
from src.core.domain.entities.ispm_monitoring import ISPMMonitoring
from src.core.domain.entities.ct_scan import CTScan
from src.core.domain.value_objects.quality_metrics import QualityMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoxelAnalysisExample:
    """
    Example class demonstrating voxel analysis workflow for PBF-LB/M research.
    """
    
    def __init__(self, output_dir: str = "voxel_analysis_output"):
        """Initialize the voxel analysis example."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.voxelizer = None
        self.fusion_system = None
        self.process_controller = None
        self.quality_analyzer = None
        
        logger.info(f"Voxel Analysis Example initialized with output directory: {self.output_dir}")
    
    def run_complete_analysis(
        self,
        cad_file_path: str,
        process_parameters: ProcessParameters,
        ispm_data: List[ISPMMonitoring],
        ct_data: List[CTScan],
        quality_metrics: List[QualityMetrics]
    ) -> Dict:
        """
        Run complete voxel analysis workflow.
        
        Args:
            cad_file_path: Path to CAD model file
            process_parameters: PBF-LB/M process parameters
            ispm_data: ISPM monitoring data
            ct_data: CT scan data
            quality_metrics: Quality metrics data
            
        Returns:
            Dict containing all analysis results
        """
        try:
            logger.info("Starting complete voxel analysis workflow...")
            
            # Step 1: Voxelize CAD model
            voxel_grid = self._voxelize_cad_model(cad_file_path, process_parameters)
            
            # Step 2: Fuse multi-modal data
            fused_data = self._fuse_multi_modal_data(voxel_grid, ispm_data, ct_data, quality_metrics)
            
            # Step 3: Perform spatial quality analysis
            spatial_metrics = self._analyze_spatial_quality(fused_data, voxel_grid)
            
            # Step 4: Generate process control recommendations
            control_actions = self._generate_process_control(voxel_grid, fused_data)
            
            # Step 5: Export results
            results = self._export_analysis_results(
                voxel_grid, fused_data, spatial_metrics, control_actions
            )
            
            logger.info("Complete voxel analysis workflow finished successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in complete voxel analysis: {e}")
            raise
    
    def _voxelize_cad_model(
        self, 
        cad_file_path: str, 
        process_parameters: ProcessParameters
    ) -> 'VoxelGrid':
        """Step 1: Voxelize CAD model with process parameters."""
        logger.info("Step 1: Voxelizing CAD model...")
        
        # Configure voxelization
        voxel_config = VoxelizationConfig(
            voxel_size=0.1,  # 0.1 mm voxels
            material_type="Ti-6Al-4V",
            coordinate_system="right_handed",
            precision=6,
            memory_limit_gb=8.0,
            parallel_processing=True,
            num_workers=4
        )
        
        # Initialize voxelizer
        self.voxelizer = CADVoxelizer(voxel_config)
        
        # Voxelize CAD model
        voxel_grid = self.voxelizer.voxelize_cad_model(cad_file_path, process_parameters)
        
        # Export voxel grid
        voxel_output_path = self.output_dir / "voxel_grid.npz"
        self.voxelizer.export_voxel_grid(voxel_grid, str(voxel_output_path))
        
        logger.info(f"CAD model voxelized: {voxel_grid.total_voxels:,} total voxels, "
                   f"{voxel_grid.solid_voxels:,} solid voxels")
        
        return voxel_grid
    
    def _fuse_multi_modal_data(
        self,
        voxel_grid: 'VoxelGrid',
        ispm_data: List[ISPMMonitoring],
        ct_data: List[CTScan],
        quality_metrics: List[QualityMetrics]
    ) -> Dict:
        """Step 2: Fuse multi-modal data with voxel grid."""
        logger.info("Step 2: Fusing multi-modal data...")
        
        # Initialize fusion system
        fusion_config = {
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
        
        self.fusion_system = MultiModalFusion(fusion_config)
        
        # Fuse data
        fused_data = self.fusion_system.fuse_voxel_data(
            voxel_grid, ispm_data, ct_data, quality_metrics
        )
        
        # Detect defects
        defect_map = self.fusion_system.detect_defects_in_voxels(fused_data)
        
        # Export fused data
        fused_output_path = self.output_dir / "fused_voxel_data.pkl"
        self.fusion_system.export_fused_data(fused_data, str(fused_output_path))
        
        logger.info(f"Multi-modal data fused: {len(fused_data)} voxels processed, "
                   f"{len(defect_map)} voxels with defects")
        
        return {
            'fused_data': fused_data,
            'defect_map': defect_map
        }
    
    def _analyze_spatial_quality(
        self, 
        fusion_results: Dict, 
        voxel_grid: 'VoxelGrid'
    ) -> 'SpatialQualityMetrics':
        """Step 3: Perform spatial quality analysis."""
        logger.info("Step 3: Analyzing spatial quality...")
        
        # Initialize quality analyzer
        quality_config = QualityAnalysisConfig(
            high_quality_threshold=90.0,
            low_quality_threshold=70.0,
            defect_threshold=0.05,
            spatial_resolution=0.1,
            cluster_min_size=10,
            gradient_smoothing=1.0,
            correlation_threshold=0.3,
            significance_level=0.05
        )
        
        self.quality_analyzer = SpatialQualityAnalyzer(quality_config)
        
        # Analyze spatial quality
        spatial_metrics = self.quality_analyzer.analyze_spatial_quality(
            fusion_results['fused_data'],
            voxel_grid.dimensions,
            voxel_grid.voxel_size
        )
        
        # Identify quality regions
        quality_regions = self.quality_analyzer.identify_quality_regions(
            fusion_results['fused_data'],
            spatial_metrics
        )
        
        # Generate quality report
        quality_report = self.quality_analyzer.generate_quality_report(
            spatial_metrics, quality_regions
        )
        
        # Export quality analysis
        quality_output_path = self.output_dir / "quality_analysis.json"
        self.quality_analyzer.export_analysis_results(
            spatial_metrics, quality_regions, str(quality_output_path)
        )
        
        # Save quality report
        report_output_path = self.output_dir / "quality_report.txt"
        with open(report_output_path, 'w') as f:
            f.write(quality_report)
        
        logger.info(f"Spatial quality analysis completed: {len(quality_regions)} quality regions identified")
        
        return spatial_metrics
    
    def _generate_process_control(
        self, 
        voxel_grid: 'VoxelGrid', 
        fusion_results: Dict
    ) -> List:
        """Step 4: Generate process control recommendations."""
        logger.info("Step 4: Generating process control recommendations...")
        
        # Configure process controller
        control_config = ProcessControlConfig(
            control_mode=ControlMode.PREDICTIVE,
            optimization_objective=OptimizationObjective.QUALITY_MAXIMIZATION,
            quality_threshold=80.0,
            defect_threshold=0.05,
            temperature_threshold=2000.0,
            max_laser_power_change=0.1,
            max_scan_speed_change=0.2,
            control_frequency=1.0,
            prediction_horizon=10,
            safety_margin=0.1
        )
        
        self.process_controller = VoxelProcessController(control_config)
        
        # Generate control actions for current layer (example: layer 5)
        current_layer = 5
        build_progress = 0.3  # 30% complete
        
        control_actions = self.process_controller.control_voxel_process(
            voxel_grid,
            fusion_results['fused_data'],
            current_layer,
            build_progress
        )
        
        # Get performance metrics
        performance_metrics = self.process_controller.get_control_performance_metrics()
        
        # Export control configuration
        control_output_path = self.output_dir / "control_config.json"
        self.process_controller.export_control_config(str(control_output_path))
        
        logger.info(f"Process control recommendations generated: {len(control_actions)} control actions")
        
        return {
            'control_actions': control_actions,
            'performance_metrics': performance_metrics
        }
    
    def _export_analysis_results(
        self,
        voxel_grid: 'VoxelGrid',
        fusion_results: Dict,
        spatial_metrics: 'SpatialQualityMetrics',
        control_results: Dict
    ) -> Dict:
        """Step 5: Export comprehensive analysis results."""
        logger.info("Step 5: Exporting analysis results...")
        
        # Create summary report
        summary_report = self._create_summary_report(
            voxel_grid, fusion_results, spatial_metrics, control_results
        )
        
        # Save summary report
        summary_output_path = self.output_dir / "analysis_summary.txt"
        with open(summary_output_path, 'w') as f:
            f.write(summary_report)
        
        # Create results dictionary
        results = {
            'voxel_grid': {
                'total_voxels': voxel_grid.total_voxels,
                'solid_voxels': voxel_grid.solid_voxels,
                'void_voxels': voxel_grid.void_voxels,
                'dimensions': voxel_grid.dimensions,
                'voxel_size': voxel_grid.voxel_size,
                'fill_ratio': voxel_grid.solid_voxels / voxel_grid.total_voxels
            },
            'fusion_results': {
                'fused_voxels': len(fusion_results['fused_data']),
                'defect_voxels': len(fusion_results['defect_map']),
                'defect_rate': len(fusion_results['defect_map']) / len(fusion_results['fused_data'])
            },
            'spatial_metrics': {
                'mean_quality': spatial_metrics.mean_quality,
                'std_quality': spatial_metrics.std_quality,
                'quality_variance': spatial_metrics.quality_variance,
                'spatial_autocorrelation': spatial_metrics.spatial_autocorrelation,
                'defect_density': spatial_metrics.defect_density,
                'geometric_deviation': spatial_metrics.geometric_deviation
            },
            'control_results': {
                'control_actions': len(control_results['control_actions']),
                'performance_metrics': control_results['performance_metrics']
            },
            'output_directory': str(self.output_dir),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Export results as JSON
        import json
        results_output_path = self.output_dir / "analysis_results.json"
        with open(results_output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Analysis results exported to: {self.output_dir}")
        
        return results
    
    def _create_summary_report(
        self,
        voxel_grid: 'VoxelGrid',
        fusion_results: Dict,
        spatial_metrics: 'SpatialQualityMetrics',
        control_results: Dict
    ) -> str:
        """Create a comprehensive summary report."""
        report = []
        report.append("=== PBF-LB/M VOXEL ANALYSIS SUMMARY REPORT ===\n")
        
        # Analysis overview
        report.append("ANALYSIS OVERVIEW:")
        report.append(f"  Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"  Output Directory: {self.output_dir}")
        report.append("")
        
        # Voxel grid summary
        report.append("VOXEL GRID SUMMARY:")
        report.append(f"  Total Voxels: {voxel_grid.total_voxels:,}")
        report.append(f"  Solid Voxels: {voxel_grid.solid_voxels:,}")
        report.append(f"  Void Voxels: {voxel_grid.void_voxels:,}")
        report.append(f"  Fill Ratio: {voxel_grid.solid_voxels/voxel_grid.total_voxels:.2%}")
        report.append(f"  Voxel Size: {voxel_grid.voxel_size} mm")
        report.append(f"  Grid Dimensions: {voxel_grid.dimensions}")
        report.append("")
        
        # Fusion results summary
        report.append("MULTI-MODAL FUSION SUMMARY:")
        report.append(f"  Fused Voxels: {len(fusion_results['fused_data']):,}")
        report.append(f"  Defect Voxels: {len(fusion_results['defect_map']):,}")
        report.append(f"  Defect Rate: {len(fusion_results['defect_map'])/len(fusion_results['fused_data']):.2%}")
        report.append("")
        
        # Quality analysis summary
        report.append("SPATIAL QUALITY ANALYSIS SUMMARY:")
        report.append(f"  Mean Quality: {spatial_metrics.mean_quality:.2f}")
        report.append(f"  Quality Standard Deviation: {spatial_metrics.std_quality:.2f}")
        report.append(f"  Quality Variance: {spatial_metrics.quality_variance:.4f}")
        report.append(f"  Spatial Autocorrelation: {spatial_metrics.spatial_autocorrelation:.4f}")
        report.append(f"  Defect Density: {spatial_metrics.defect_density:.4f}")
        report.append(f"  Geometric Deviation: {spatial_metrics.geometric_deviation:.4f}")
        report.append("")
        
        # Process control summary
        report.append("PROCESS CONTROL SUMMARY:")
        report.append(f"  Control Actions Generated: {len(control_results['control_actions'])}")
        report.append(f"  Control Mode: {control_results['performance_metrics'].get('control_mode', 'N/A')}")
        report.append(f"  Optimization Objective: {control_results['performance_metrics'].get('optimization_objective', 'N/A')}")
        report.append("")
        
        # Key findings
        report.append("KEY FINDINGS:")
        if spatial_metrics.mean_quality > 90:
            report.append("  ✓ High overall quality achieved")
        elif spatial_metrics.mean_quality > 80:
            report.append("  ⚠ Moderate quality - room for improvement")
        else:
            report.append("  ✗ Low quality - significant improvements needed")
        
        if spatial_metrics.defect_density < 0.01:
            report.append("  ✓ Low defect density")
        elif spatial_metrics.defect_density < 0.05:
            report.append("  ⚠ Moderate defect density")
        else:
            report.append("  ✗ High defect density - attention required")
        
        if spatial_metrics.spatial_autocorrelation > 0.3:
            report.append("  ✓ Good spatial quality consistency")
        else:
            report.append("  ⚠ Spatial quality variations detected")
        
        report.append("")
        report.append("=== END OF SUMMARY REPORT ===")
        
        return "\n".join(report)
    
    def create_sample_data(self) -> tuple:
        """Create sample data for demonstration purposes."""
        logger.info("Creating sample data for demonstration...")
        
        # Sample process parameters
        process_params = ProcessParameters(
            laser_power=300.0,  # Watts
            scan_speed=1000.0,  # mm/s
            layer_thickness=0.03,  # mm
            hatch_spacing=0.1,  # mm
            build_temperature=80.0,  # Celsius
            chamber_temperature=40.0,  # Celsius
            oxygen_content=0.1,  # %
            build_time=3600.0  # seconds
        )
        
        # Sample ISPM data
        ispm_data = []
        for i in range(100):
            ispm_point = ISPMMonitoring(
                timestamp=datetime.now(),
                sensor_type="temperature",
                measurement_value=1500.0 + np.random.normal(0, 50),
                unit="Celsius",
                position=(np.random.uniform(0, 10), np.random.uniform(0, 10), np.random.uniform(0, 5)),
                confidence=0.9
            )
            ispm_data.append(ispm_point)
        
        # Sample CT data
        ct_data = []
        for i in range(200):
            ct_point = CTScan(
                position=(np.random.uniform(0, 10), np.random.uniform(0, 10), np.random.uniform(0, 5)),
                density=4.43 + np.random.normal(0, 0.1),  # Ti-6Al-4V density
                intensity=1000 + np.random.normal(0, 100),
                defect_probability=np.random.uniform(0, 0.1),
                material_type="Ti-6Al-4V",
                quality_score=90.0 + np.random.normal(0, 10)
            )
            ct_data.append(ct_point)
        
        # Sample quality metrics
        quality_metrics = []
        for i in range(50):
            quality_point = QualityMetrics(
                quality_score=85.0 + np.random.normal(0, 15),
                dimensional_accuracy=0.05 + np.random.uniform(0, 0.02),
                surface_roughness_ra=5.0 + np.random.uniform(0, 2.0),
                defect_count=np.random.poisson(2),
                defect_types=["porosity", "crack"] if np.random.random() > 0.8 else []
            )
            quality_metrics.append(quality_point)
        
        logger.info("Sample data created successfully")
        return process_params, ispm_data, ct_data, quality_metrics


def main():
    """Main function to run the voxel analysis example."""
    try:
        # Initialize example
        example = VoxelAnalysisExample("voxel_analysis_demo")
        
        # Create sample data
        process_params, ispm_data, ct_data, quality_metrics = example.create_sample_data()
        
        # Note: In a real scenario, you would provide actual CAD file path
        # For this example, we'll use a placeholder
        cad_file_path = "sample_part.stl"  # Replace with actual CAD file
        
        logger.info("Note: This example uses sample data. In production, provide actual CAD files and sensor data.")
        
        # Run complete analysis (commented out as it requires actual CAD file)
        # results = example.run_complete_analysis(
        #     cad_file_path, process_params, ispm_data, ct_data, quality_metrics
        # )
        
        # Print sample data information
        logger.info("Sample data created:")
        logger.info(f"  Process Parameters: {process_params}")
        logger.info(f"  ISPM Data Points: {len(ispm_data)}")
        logger.info(f"  CT Data Points: {len(ct_data)}")
        logger.info(f"  Quality Metrics: {len(quality_metrics)}")
        
        logger.info("Voxel analysis example completed successfully!")
        logger.info("To run with actual data, provide a CAD file path and uncomment the analysis call.")
        
    except Exception as e:
        logger.error(f"Error in voxel analysis example: {e}")
        raise


if __name__ == "__main__":
    main()
