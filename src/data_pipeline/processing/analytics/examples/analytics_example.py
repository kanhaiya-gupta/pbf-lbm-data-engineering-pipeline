"""
Comprehensive Analytics Example for PBF-LB/M Data Pipeline

This example demonstrates the complete analytics capabilities of the PBF-LB/M
data pipeline, including sensitivity analysis, statistical analysis, and
process analysis for additive manufacturing research.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime, timedelta

# Import analytics components
from ..sensitivity_analysis.global_analysis import GlobalSensitivityAnalyzer, SobolAnalyzer, MorrisAnalyzer
from ..sensitivity_analysis.doe import ExperimentalDesigner, FactorialDesign
from ..sensitivity_analysis.uncertainty import UncertaintyQuantifier, MonteCarloAnalyzer
from ..statistical_analysis.multivariate import MultivariateAnalyzer, PCAAnalyzer
from ..statistical_analysis.time_series import TimeSeriesAnalyzer, TrendAnalyzer

logger = logging.getLogger(__name__)


class PBFProcessModel:
    """
    Simplified PBF-LB/M process model for demonstration.
    
    This model simulates the relationship between process parameters and
    quality outcomes in PBF-LB/M additive manufacturing.
    """
    
    def __init__(self):
        """Initialize the PBF process model."""
        self.parameter_names = [
            'laser_power',      # Laser power (W)
            'scan_speed',       # Scan speed (mm/s)
            'hatch_spacing',    # Hatch spacing (mm)
            'layer_thickness',  # Layer thickness (mm)
            'preheat_temp',     # Preheat temperature (°C)
            'atmosphere_pressure'  # Atmosphere pressure (mbar)
        ]
        
        self.parameter_bounds = {
            'laser_power': (100, 400),           # 100-400 W
            'scan_speed': (500, 2000),           # 500-2000 mm/s
            'hatch_spacing': (0.05, 0.15),       # 0.05-0.15 mm
            'layer_thickness': (0.02, 0.08),     # 0.02-0.08 mm
            'preheat_temp': (80, 200),           # 80-200 °C
            'atmosphere_pressure': (0.1, 1.0)    # 0.1-1.0 mbar
        }
    
    def evaluate_quality(self, parameters: np.ndarray) -> float:
        """
        Evaluate process quality based on parameters.
        
        This is a simplified model that simulates the relationship between
        process parameters and quality outcomes.
        """
        laser_power, scan_speed, hatch_spacing, layer_thickness, preheat_temp, atmosphere_pressure = parameters
        
        # Simulate quality as a function of process parameters
        # Higher laser power generally improves quality (up to a point)
        power_effect = 0.3 * (laser_power / 300) * np.exp(-(laser_power - 300)**2 / (2 * 50**2))
        
        # Optimal scan speed around 1000 mm/s
        speed_effect = 0.2 * np.exp(-(scan_speed - 1000)**2 / (2 * 300**2))
        
        # Smaller hatch spacing improves quality
        hatch_effect = 0.15 * (0.15 - hatch_spacing) / 0.1
        
        # Thinner layers generally improve quality
        layer_effect = 0.1 * (0.08 - layer_thickness) / 0.06
        
        # Higher preheat temperature improves quality
        temp_effect = 0.1 * (preheat_temp - 80) / 120
        
        # Lower atmosphere pressure improves quality
        pressure_effect = 0.1 * (1.0 - atmosphere_pressure) / 0.9
        
        # Add some noise
        noise = np.random.normal(0, 0.05)
        
        # Calculate overall quality (0-1 scale)
        quality = (power_effect + speed_effect + hatch_effect + 
                  layer_effect + temp_effect + pressure_effect + noise)
        
        # Ensure quality is between 0 and 1
        quality = np.clip(quality, 0, 1)
        
        return quality
    
    def evaluate_porosity(self, parameters: np.ndarray) -> float:
        """
        Evaluate porosity based on parameters.
        
        Lower porosity is better (closer to 0).
        """
        laser_power, scan_speed, hatch_spacing, layer_thickness, preheat_temp, atmosphere_pressure = parameters
        
        # Simulate porosity as a function of process parameters
        # Higher laser power reduces porosity
        power_effect = -0.2 * (laser_power / 300)
        
        # Optimal scan speed reduces porosity
        speed_effect = -0.15 * np.exp(-(scan_speed - 1000)**2 / (2 * 400**2))
        
        # Larger hatch spacing increases porosity
        hatch_effect = 0.3 * hatch_spacing / 0.15
        
        # Thicker layers increase porosity
        layer_effect = 0.2 * layer_thickness / 0.08
        
        # Higher preheat temperature reduces porosity
        temp_effect = -0.1 * (preheat_temp - 80) / 120
        
        # Higher atmosphere pressure increases porosity
        pressure_effect = 0.2 * atmosphere_pressure / 1.0
        
        # Add some noise
        noise = np.random.normal(0, 0.02)
        
        # Calculate porosity (0-1 scale)
        porosity = (power_effect + speed_effect + hatch_effect + 
                   layer_effect + temp_effect + pressure_effect + noise)
        
        # Ensure porosity is between 0 and 1
        porosity = np.clip(porosity, 0, 1)
        
        return porosity


class AnalyticsExample:
    """
    Comprehensive analytics example for PBF-LB/M data pipeline.
    
    This class demonstrates the complete analytics workflow including
    sensitivity analysis, statistical analysis, and process optimization.
    """
    
    def __init__(self):
        """Initialize the analytics example."""
        self.process_model = PBFProcessModel()
        self.analyzers = self._initialize_analyzers()
        self.results = {}
        
        logger.info("Analytics Example initialized")
    
    def _initialize_analyzers(self) -> Dict[str, Any]:
        """Initialize all analytics components."""
        return {
            'global_sensitivity': GlobalSensitivityAnalyzer(),
            'sobol': SobolAnalyzer(),
            'morris': MorrisAnalyzer(),
            'experimental_design': ExperimentalDesigner(),
            'uncertainty': UncertaintyQuantifier(),
            'monte_carlo': MonteCarloAnalyzer(),
            'multivariate': MultivariateAnalyzer(),
            'pca': PCAAnalyzer(),
            'time_series': TimeSeriesAnalyzer(),
            'trend': TrendAnalyzer()
        }
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run complete analytics workflow.
        
        Returns:
            Dict containing all analysis results
        """
        logger.info("Starting complete analytics workflow")
        
        # 1. Generate experimental design
        logger.info("Step 1: Generating experimental design")
        experimental_design = self._generate_experimental_design()
        
        # 2. Run sensitivity analysis
        logger.info("Step 2: Running sensitivity analysis")
        sensitivity_results = self._run_sensitivity_analysis()
        
        # 3. Run uncertainty quantification
        logger.info("Step 3: Running uncertainty quantification")
        uncertainty_results = self._run_uncertainty_analysis()
        
        # 4. Generate synthetic data
        logger.info("Step 4: Generating synthetic process data")
        process_data = self._generate_synthetic_data(experimental_design)
        
        # 5. Run statistical analysis
        logger.info("Step 5: Running statistical analysis")
        statistical_results = self._run_statistical_analysis(process_data)
        
        # 6. Run time series analysis
        logger.info("Step 6: Running time series analysis")
        time_series_results = self._run_time_series_analysis(process_data)
        
        # 7. Compile results
        logger.info("Step 7: Compiling results")
        self.results = {
            'experimental_design': experimental_design,
            'sensitivity_analysis': sensitivity_results,
            'uncertainty_analysis': uncertainty_results,
            'process_data': process_data,
            'statistical_analysis': statistical_results,
            'time_series_analysis': time_series_results,
            'analysis_timestamp': datetime.now()
        }
        
        logger.info("Complete analytics workflow finished")
        return self.results
    
    def _generate_experimental_design(self) -> Dict[str, Any]:
        """Generate experimental design for PBF process."""
        # Create factorial design
        factorial_design = self.analyzers['experimental_design'].create_factorial_design(
            parameter_bounds=self.process_model.parameter_bounds,
            parameter_names=self.process_model.parameter_names,
            levels=2
        )
        
        # Create response surface design
        response_surface_design = self.analyzers['experimental_design'].create_response_surface_design(
            parameter_bounds=self.process_model.parameter_bounds,
            parameter_names=self.process_model.parameter_names,
            design_type="ccd"
        )
        
        return {
            'factorial_design': factorial_design,
            'response_surface_design': response_surface_design
        }
    
    def _run_sensitivity_analysis(self) -> Dict[str, Any]:
        """Run sensitivity analysis on PBF process model."""
        # Sobol analysis
        sobol_result = self.analyzers['sobol'].analyze(
            model_function=self.process_model.evaluate_quality,
            parameter_bounds=self.process_model.parameter_bounds,
            parameter_names=self.process_model.parameter_names
        )
        
        # Morris screening
        morris_result = self.analyzers['morris'].analyze(
            model_function=self.process_model.evaluate_quality,
            parameter_bounds=self.process_model.parameter_bounds,
            parameter_names=self.process_model.parameter_names
        )
        
        return {
            'sobol_analysis': sobol_result,
            'morris_analysis': morris_result
        }
    
    def _run_uncertainty_analysis(self) -> Dict[str, Any]:
        """Run uncertainty quantification analysis."""
        # Define parameter distributions
        parameter_distributions = {
            'laser_power': {'type': 'normal', 'params': {'mu': 250, 'sigma': 50}},
            'scan_speed': {'type': 'normal', 'params': {'mu': 1000, 'sigma': 200}},
            'hatch_spacing': {'type': 'uniform', 'params': {'lower': 0.05, 'upper': 0.15}},
            'layer_thickness': {'type': 'uniform', 'params': {'lower': 0.02, 'upper': 0.08}},
            'preheat_temp': {'type': 'normal', 'params': {'mu': 120, 'sigma': 30}},
            'atmosphere_pressure': {'type': 'uniform', 'params': {'lower': 0.1, 'upper': 1.0}}
        }
        
        # Monte Carlo analysis
        monte_carlo_result = self.analyzers['monte_carlo'].analyze(
            model_function=self.process_model.evaluate_quality,
            parameter_distributions=parameter_distributions,
            parameter_names=self.process_model.parameter_names
        )
        
        return {
            'monte_carlo_analysis': monte_carlo_result
        }
    
    def _generate_synthetic_data(self, experimental_design: Dict[str, Any]) -> pd.DataFrame:
        """Generate synthetic process data based on experimental design."""
        # Use factorial design for data generation
        design_matrix = experimental_design['factorial_design'].design_matrix
        
        # Generate synthetic data
        data_rows = []
        for _, row in design_matrix.iterrows():
            parameters = row[self.process_model.parameter_names].values
            
            # Evaluate model
            quality = self.process_model.evaluate_quality(parameters)
            porosity = self.process_model.evaluate_porosity(parameters)
            
            # Add some additional synthetic features
            density = 1.0 - porosity  # Density is inverse of porosity
            surface_roughness = 0.1 + 0.05 * np.random.random()  # Random surface roughness
            tensile_strength = 400 + 200 * quality + 50 * np.random.random()  # Tensile strength
            
            data_row = {
                **row.to_dict(),
                'quality': quality,
                'porosity': porosity,
                'density': density,
                'surface_roughness': surface_roughness,
                'tensile_strength': tensile_strength
            }
            data_rows.append(data_row)
        
        return pd.DataFrame(data_rows)
    
    def _run_statistical_analysis(self, process_data: pd.DataFrame) -> Dict[str, Any]:
        """Run statistical analysis on process data."""
        # Define feature names
        feature_names = self.process_model.parameter_names
        
        # PCA analysis
        pca_result = self.analyzers['pca'].analyze(
            data=process_data,
            feature_names=feature_names,
            n_components=3
        )
        
        # Clustering analysis
        clustering_result = self.analyzers['multivariate'].analyze_clustering(
            data=process_data,
            feature_names=feature_names,
            method="kmeans",
            n_clusters=3
        )
        
        # Correlation analysis
        correlation_result = self.analyzers['multivariate'].analyze_correlation(
            data=process_data,
            feature_names=feature_names + ['quality', 'porosity', 'density']
        )
        
        return {
            'pca_analysis': pca_result,
            'clustering_analysis': clustering_result,
            'correlation_analysis': correlation_result
        }
    
    def _run_time_series_analysis(self, process_data: pd.DataFrame) -> Dict[str, Any]:
        """Run time series analysis on process data."""
        # Create synthetic time series data
        n_points = 100
        time_index = pd.date_range(start='2024-01-01', periods=n_points, freq='H')
        
        # Generate synthetic time series for quality
        quality_series = pd.Series(
            data=0.7 + 0.2 * np.sin(np.linspace(0, 4*np.pi, n_points)) + 0.1 * np.random.random(n_points),
            index=time_index,
            name='quality'
        )
        
        # Trend analysis
        trend_result = self.analyzers['trend'].analyze(
            time_series=quality_series,
            method="linear"
        )
        
        # Seasonality analysis
        seasonality_result = self.analyzers['time_series'].analyze_seasonality(quality_series)
        
        # Forecasting
        forecasting_result = self.analyzers['time_series'].analyze_forecasting(
            time_series=quality_series,
            forecast_horizon=10
        )
        
        return {
            'trend_analysis': trend_result,
            'seasonality_analysis': seasonality_result,
            'forecasting_analysis': forecasting_result,
            'time_series_data': quality_series
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive analytics report."""
        if not self.results:
            return "No analysis results available. Run complete_analysis() first."
        
        report = []
        report.append("=" * 80)
        report.append("PBF-LB/M DATA PIPELINE ANALYTICS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {self.results['analysis_timestamp']}")
        report.append("")
        
        # Experimental Design Results
        report.append("EXPERIMENTAL DESIGN")
        report.append("-" * 40)
        factorial_design = self.results['experimental_design']['factorial_design']
        report.append(f"Factorial Design: {factorial_design.design_points} points")
        report.append(f"Design Type: {factorial_design.design_type}")
        report.append(f"Design Quality: {factorial_design.design_quality}")
        report.append("")
        
        # Sensitivity Analysis Results
        report.append("SENSITIVITY ANALYSIS")
        report.append("-" * 40)
        sobol_result = self.results['sensitivity_analysis']['sobol_analysis']
        if sobol_result.success:
            report.append("Sobol Analysis Results:")
            for param, value in sobol_result.sensitivity_indices.items():
                if param.startswith('S1_'):
                    report.append(f"  {param}: {value:.4f}")
        report.append("")
        
        # Uncertainty Analysis Results
        report.append("UNCERTAINTY ANALYSIS")
        report.append("-" * 40)
        monte_carlo_result = self.results['uncertainty_analysis']['monte_carlo_analysis']
        if monte_carlo_result.success:
            output_stats = monte_carlo_result.output_statistics
            report.append(f"Output Mean: {output_stats['mean']:.4f}")
            report.append(f"Output Std: {output_stats['std']:.4f}")
            report.append(f"Output Min: {output_stats['min']:.4f}")
            report.append(f"Output Max: {output_stats['max']:.4f}")
        report.append("")
        
        # Statistical Analysis Results
        report.append("STATISTICAL ANALYSIS")
        report.append("-" * 40)
        pca_result = self.results['statistical_analysis']['pca_analysis']
        if pca_result.success:
            explained_variance = pca_result.explained_variance
            report.append(f"PCA Components: {pca_result.analysis_results['n_components']}")
            report.append(f"Total Variance Explained: {explained_variance['total_variance_explained']:.4f}")
        report.append("")
        
        # Time Series Analysis Results
        report.append("TIME SERIES ANALYSIS")
        report.append("-" * 40)
        trend_result = self.results['time_series_analysis']['trend_analysis']
        if trend_result.success:
            trend_analysis = trend_result.trend_analysis
            report.append(f"Trend Direction: {trend_analysis.get('trend_direction', 'unknown')}")
            report.append(f"R-squared: {trend_analysis.get('r_squared', 0):.4f}")
        report.append("")
        
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_results(self, filename: str = None):
        """Save analysis results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pbf_analytics_results_{timestamp}.json"
        
        # Convert results to JSON-serializable format
        import json
        
        serializable_results = {}
        for key, value in self.results.items():
            if key == 'analysis_timestamp':
                serializable_results[key] = value.isoformat()
            elif hasattr(value, '__dict__'):
                serializable_results[key] = str(value)
            else:
                serializable_results[key] = value
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filename}")


def main():
    """Main function to run the analytics example."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create analytics example
    example = AnalyticsExample()
    
    # Run complete analysis
    results = example.run_complete_analysis()
    
    # Generate and print report
    report = example.generate_report()
    print(report)
    
    # Save results
    example.save_results()
    
    print("\nAnalytics example completed successfully!")
    print("Check the generated JSON file for detailed results.")


if __name__ == "__main__":
    main()



