"""
Documentation for PBF-LB/M Analytics

This module provides comprehensive documentation capabilities for PBF-LB/M
analytics, including API documentation, user guides, and technical
documentation generation.
"""

import os
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from datetime import datetime
import json
import warnings

logger = logging.getLogger(__name__)


@dataclass
class DocumentationConfig:
    """Configuration for documentation generation."""
    
    # Documentation parameters
    doc_format: str = "markdown"  # "markdown", "html", "rst"
    include_examples: bool = True
    include_api_reference: bool = True
    
    # Output parameters
    output_directory: str = "docs"
    filename_prefix: str = "pbf_analytics_docs"
    
    # Content parameters
    include_installation: bool = True
    include_quick_start: bool = True
    include_tutorials: bool = True


@dataclass
class DocumentationResult:
    """Result of documentation generation."""
    
    success: bool
    doc_type: str
    doc_path: str
    doc_size: int
    generation_time: float
    error_message: Optional[str] = None


class AnalysisDocumentation:
    """
    Analysis documentation generator for PBF-LB/M analytics.
    
    This class provides comprehensive documentation generation capabilities
    including API documentation, user guides, and technical documentation
    for PBF-LB/M analytics.
    """
    
    def __init__(self, config: DocumentationConfig = None):
        """Initialize the documentation generator."""
        self.config = config or DocumentationConfig()
        
        # Create output directory
        os.makedirs(self.config.output_directory, exist_ok=True)
        
        logger.info("Analysis Documentation Generator initialized")
    
    def generate_user_guide(
        self,
        doc_title: str = "PBF-LB/M Analytics User Guide"
    ) -> DocumentationResult:
        """
        Generate user guide documentation.
        
        Args:
            doc_title: Title of the documentation
            
        Returns:
            DocumentationResult: Documentation generation result
        """
        try:
            start_time = datetime.now()
            
            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config.filename_prefix}_user_guide_{timestamp}.{self.config.doc_format}"
            doc_path = os.path.join(self.config.output_directory, filename)
            
            # Generate documentation content
            if self.config.doc_format == "markdown":
                doc_content = self._generate_user_guide_markdown(doc_title)
            elif self.config.doc_format == "html":
                doc_content = self._generate_user_guide_html(doc_title)
            else:
                raise ValueError(f"Unsupported documentation format: {self.config.doc_format}")
            
            # Write documentation to file
            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write(doc_content)
            
            # Calculate documentation size
            doc_size = os.path.getsize(doc_path)
            
            # Calculate generation time
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = DocumentationResult(
                success=True,
                doc_type="UserGuide",
                doc_path=doc_path,
                doc_size=doc_size,
                generation_time=generation_time
            )
            
            logger.info(f"User guide documentation generated: {doc_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating user guide documentation: {e}")
            return DocumentationResult(
                success=False,
                doc_type="UserGuide",
                doc_path="",
                doc_size=0,
                generation_time=0.0,
                error_message=str(e)
            )
    
    def generate_api_documentation(
        self,
        doc_title: str = "PBF-LB/M Analytics API Documentation"
    ) -> DocumentationResult:
        """
        Generate API documentation.
        
        Args:
            doc_title: Title of the documentation
            
        Returns:
            DocumentationResult: Documentation generation result
        """
        try:
            start_time = datetime.now()
            
            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config.filename_prefix}_api_{timestamp}.{self.config.doc_format}"
            doc_path = os.path.join(self.config.output_directory, filename)
            
            # Generate documentation content
            if self.config.doc_format == "markdown":
                doc_content = self._generate_api_documentation_markdown(doc_title)
            elif self.config.doc_format == "html":
                doc_content = self._generate_api_documentation_html(doc_title)
            else:
                raise ValueError(f"Unsupported documentation format: {self.config.doc_format}")
            
            # Write documentation to file
            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write(doc_content)
            
            # Calculate documentation size
            doc_size = os.path.getsize(doc_path)
            
            # Calculate generation time
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = DocumentationResult(
                success=True,
                doc_type="APIDocumentation",
                doc_path=doc_path,
                doc_size=doc_size,
                generation_time=generation_time
            )
            
            logger.info(f"API documentation generated: {doc_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating API documentation: {e}")
            return DocumentationResult(
                success=False,
                doc_type="APIDocumentation",
                doc_path="",
                doc_size=0,
                generation_time=0.0,
                error_message=str(e)
            )
    
    def _generate_user_guide_markdown(self, doc_title: str) -> str:
        """Generate user guide in Markdown format."""
        markdown_content = f"""# {doc_title}

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Sensitivity Analysis](#sensitivity-analysis)
4. [Statistical Analysis](#statistical-analysis)
5. [Process Analysis](#process-analysis)
6. [Reporting](#reporting)
7. [Examples](#examples)
8. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
pip install SALib  # For sensitivity analysis
pip install pymc3  # For Bayesian analysis (optional)
```

### Install PBF-LB/M Analytics

```bash
pip install pbf-lbm-analytics
```

## Quick Start

### Basic Usage

```python
from src.data_pipeline.processing.analytics import GlobalSensitivityAnalyzer

# Initialize analyzer
analyzer = GlobalSensitivityAnalyzer()

# Define your model function
def my_model(parameters):
    # Your PBF-LB/M model here
    return quality_score

# Define parameter bounds
parameter_bounds = {
    'laser_power': (100, 400),
    'scan_speed': (500, 2000),
    'hatch_spacing': (0.05, 0.15)
}

# Run sensitivity analysis
result = analyzer.analyze_sobol(my_model, parameter_bounds)

# Print results
print(f"Analysis successful: {result.success}")
print(f"Analysis time: {result.analysis_time:.2f}s")
```

## Sensitivity Analysis

### Sobol Analysis

Sobol analysis provides global sensitivity indices for understanding parameter influences.

```python
from src.data_pipeline.processing.analytics import SobolAnalyzer

analyzer = SobolAnalyzer()
result = analyzer.analyze(model_function, parameter_bounds)

# Access sensitivity indices
for param, value in result.sensitivity_indices.items():
    if param.startswith('S1_'):
        print(f"{param}: {value:.4f}")
```

### Morris Screening

Morris screening provides efficient parameter screening for high-dimensional problems.

```python
from src.data_pipeline.processing.analytics import MorrisAnalyzer

analyzer = MorrisAnalyzer()
result = analyzer.analyze(model_function, parameter_bounds)

# Access Morris indices
for param, value in result.sensitivity_indices.items():
    if param.startswith('mu_star_'):
        print(f"{param}: {value:.4f}")
```

## Statistical Analysis

### Multivariate Analysis

```python
from src.data_pipeline.processing.analytics import MultivariateAnalyzer

analyzer = MultivariateAnalyzer()
result = analyzer.analyze_pca(data, feature_names)

# Access PCA results
print(f"Explained variance: {result.explained_variance['total_variance_explained']:.4f}")
```

### Time Series Analysis

```python
from src.data_pipeline.processing.analytics import TimeSeriesAnalyzer

analyzer = TimeSeriesAnalyzer()
result = analyzer.analyze_trend(time_series_data)

# Access trend results
print(f"Trend direction: {result.trend_analysis['trend_direction']}")
```

## Process Analysis

### Parameter Analysis

```python
from src.data_pipeline.processing.analytics import ParameterAnalyzer

analyzer = ParameterAnalyzer()
result = analyzer.analyze_parameter_optimization(
    objective_function, parameter_bounds
)

# Access optimal parameters
print(f"Optimal parameters: {result.optimal_parameters}")
```

### Quality Analysis

```python
from src.data_pipeline.processing.analytics import QualityAnalyzer

analyzer = QualityAnalyzer()
result = analyzer.analyze_quality_prediction(process_data, quality_target)

# Access quality metrics
print(f"Mean quality: {result.quality_metrics['mean_quality']:.4f}")
```

## Reporting

### Generate Reports

```python
from src.data_pipeline.processing.analytics import AnalysisReportGenerator

generator = AnalysisReportGenerator()
result = generator.generate_comprehensive_report(analytics_results)

print(f"Report generated: {result.report_path}")
```

### Generate Visualizations

```python
from src.data_pipeline.processing.analytics import AnalysisVisualizer

visualizer = AnalysisVisualizer()
result = visualizer.visualize_sensitivity_analysis(sensitivity_results)

print(f"Plots generated: {result.plot_paths}")
```

## Examples

### Complete Workflow Example

```python
from src.data_pipeline.processing.analytics.examples import AnalyticsExample

# Create example
example = AnalyticsExample()

# Run complete analysis
results = example.run_complete_analysis()

# Generate report
report = example.generate_report()
print(report)

# Save results
example.save_results()
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'SALib'**
   - Solution: Install SALib with `pip install SALib`

2. **MemoryError during analysis**
   - Solution: Reduce sample size in configuration

3. **Analysis takes too long**
   - Solution: Use Morris screening instead of Sobol for initial analysis

### Getting Help

- Check the API documentation for detailed method descriptions
- Review the examples in the `examples/` directory
- Submit issues on the project repository

## License

This software is licensed under the MIT License.

## Citation

If you use this software in your research, please cite:

```
PBF-LB/M Analytics: Advanced Analytics for Additive Manufacturing
[Your Citation Here]
```
"""
        return markdown_content
    
    def _generate_user_guide_html(self, doc_title: str) -> str:
        """Generate user guide in HTML format."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{doc_title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; }}
        h3 {{ color: #7f8c8d; }}
        code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
        pre {{ background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        .toc {{ background-color: #ecf0f1; padding: 20px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>{doc_title}</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <div class="toc">
        <h2>Table of Contents</h2>
        <ul>
            <li><a href="#installation">Installation</a></li>
            <li><a href="#quick-start">Quick Start</a></li>
            <li><a href="#sensitivity-analysis">Sensitivity Analysis</a></li>
            <li><a href="#statistical-analysis">Statistical Analysis</a></li>
            <li><a href="#process-analysis">Process Analysis</a></li>
            <li><a href="#reporting">Reporting</a></li>
            <li><a href="#examples">Examples</a></li>
            <li><a href="#troubleshooting">Troubleshooting</a></li>
        </ul>
    </div>
    
    <h2 id="installation">Installation</h2>
    <h3>Prerequisites</h3>
    <ul>
        <li>Python 3.8 or higher</li>
        <li>pip package manager</li>
    </ul>
    
    <h3>Install Dependencies</h3>
    <pre><code>pip install numpy pandas scipy scikit-learn matplotlib seaborn
pip install SALib  # For sensitivity analysis
pip install pymc3  # For Bayesian analysis (optional)</code></pre>
    
    <h2 id="quick-start">Quick Start</h2>
    <h3>Basic Usage</h3>
    <pre><code>from src.data_pipeline.processing.analytics import GlobalSensitivityAnalyzer

# Initialize analyzer
analyzer = GlobalSensitivityAnalyzer()

# Define your model function
def my_model(parameters):
    # Your PBF-LB/M model here
    return quality_score

# Define parameter bounds
parameter_bounds = {{
    'laser_power': (100, 400),
    'scan_speed': (500, 2000),
    'hatch_spacing': (0.05, 0.15)
}}

# Run sensitivity analysis
result = analyzer.analyze_sobol(my_model, parameter_bounds)

# Print results
print(f"Analysis successful: {{result.success}}")
print(f"Analysis time: {{result.analysis_time:.2f}}s")</code></pre>
    
    <h2 id="sensitivity-analysis">Sensitivity Analysis</h2>
    <p>Sensitivity analysis helps understand which parameters have the most influence on your PBF-LB/M process outcomes.</p>
    
    <h2 id="statistical-analysis">Statistical Analysis</h2>
    <p>Statistical analysis provides insights into data patterns and relationships.</p>
    
    <h2 id="process-analysis">Process Analysis</h2>
    <p>Process analysis focuses on optimizing PBF-LB/M process parameters and quality outcomes.</p>
    
    <h2 id="reporting">Reporting</h2>
    <p>Generate comprehensive reports and visualizations of your analysis results.</p>
    
    <h2 id="examples">Examples</h2>
    <p>See the examples directory for complete workflow demonstrations.</p>
    
    <h2 id="troubleshooting">Troubleshooting</h2>
    <p>Common issues and solutions for using PBF-LB/M Analytics.</p>
</body>
</html>
"""
        return html_content
    
    def _generate_api_documentation_markdown(self, doc_title: str) -> str:
        """Generate API documentation in Markdown format."""
        markdown_content = f"""# {doc_title}

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## API Reference

### Sensitivity Analysis

#### GlobalSensitivityAnalyzer

Main class for global sensitivity analysis.

```python
class GlobalSensitivityAnalyzer:
    def __init__(self, config: SensitivityConfig = None):
        \"\"\"Initialize the global sensitivity analyzer.\"\"\"
    
    def analyze_sobol(self, model_function, parameter_bounds, parameter_names=None):
        \"\"\"Perform Sobol sensitivity analysis.\"\"\"
    
    def analyze_morris(self, model_function, parameter_bounds, parameter_names=None):
        \"\"\"Perform Morris screening analysis.\"\"\"
```

#### SobolAnalyzer

Specialized Sobol sensitivity analyzer.

```python
class SobolAnalyzer(GlobalSensitivityAnalyzer):
    def analyze(self, model_function, parameter_bounds, parameter_names=None):
        \"\"\"Perform Sobol analysis.\"\"\"
```

#### MorrisAnalyzer

Specialized Morris sensitivity analyzer.

```python
class MorrisAnalyzer(GlobalSensitivityAnalyzer):
    def analyze(self, model_function, parameter_bounds, parameter_names=None):
        \"\"\"Perform Morris analysis.\"\"\"
```

### Statistical Analysis

#### MultivariateAnalyzer

Main class for multivariate analysis.

```python
class MultivariateAnalyzer:
    def __init__(self, config: MultivariateConfig = None):
        \"\"\"Initialize the multivariate analyzer.\"\"\"
    
    def analyze_pca(self, data, feature_names=None, n_components=None):
        \"\"\"Perform principal component analysis.\"\"\"
    
    def analyze_clustering(self, data, feature_names=None, method=None, n_clusters=None):
        \"\"\"Perform clustering analysis.\"\"\"
```

#### TimeSeriesAnalyzer

Main class for time series analysis.

```python
class TimeSeriesAnalyzer:
    def __init__(self, config: TimeSeriesConfig = None):
        \"\"\"Initialize the time series analyzer.\"\"\"
    
    def analyze_trend(self, time_series, method=None):
        \"\"\"Perform trend analysis.\"\"\"
    
    def analyze_seasonality(self, time_series):
        \"\"\"Perform seasonality analysis.\"\"\"
```

### Process Analysis

#### ParameterAnalyzer

Main class for parameter analysis.

```python
class ParameterAnalyzer:
    def __init__(self, config: ParameterAnalysisConfig = None):
        \"\"\"Initialize the parameter analyzer.\"\"\"
    
    def analyze_parameter_optimization(self, objective_function, parameter_bounds, parameter_names=None):
        \"\"\"Perform parameter optimization analysis.\"\"\"
    
    def analyze_parameter_interactions(self, process_data, parameter_names=None, target_variable=None):
        \"\"\"Analyze parameter interactions in process data.\"\"\"
```

#### QualityAnalyzer

Main class for quality analysis.

```python
class QualityAnalyzer:
    def __init__(self, config: QualityAnalysisConfig = None):
        \"\"\"Initialize the quality analyzer.\"\"\"
    
    def analyze_quality_prediction(self, process_data, quality_target, feature_names=None):
        \"\"\"Perform quality prediction analysis.\"\"\"
```

### Reporting

#### AnalysisReportGenerator

Main class for report generation.

```python
class AnalysisReportGenerator:
    def __init__(self, config: ReportConfig = None):
        \"\"\"Initialize the report generator.\"\"\"
    
    def generate_comprehensive_report(self, analytics_results, report_title="PBF-LB/M Analytics Report"):
        \"\"\"Generate comprehensive analytics report.\"\"\"
    
    def generate_sensitivity_report(self, sensitivity_results, report_title="Sensitivity Analysis Report"):
        \"\"\"Generate sensitivity analysis report.\"\"\"
```

#### AnalysisVisualizer

Main class for visualization generation.

```python
class AnalysisVisualizer:
    def __init__(self, config: VisualizationConfig = None):
        \"\"\"Initialize the visualizer.\"\"\"
    
    def visualize_sensitivity_analysis(self, sensitivity_results, plot_title="Sensitivity Analysis"):
        \"\"\"Visualize sensitivity analysis results.\"\"\"
    
    def visualize_statistical_analysis(self, statistical_results, plot_title="Statistical Analysis"):
        \"\"\"Visualize statistical analysis results.\"\"\"
```

## Configuration Classes

### SensitivityConfig

Configuration for sensitivity analysis.

```python
@dataclass
class SensitivityConfig:
    sample_size: int = 1000
    confidence_level: float = 0.95
    random_seed: Optional[int] = None
    sobol_order: int = 2
    morris_levels: int = 10
    morris_num_trajectories: int = 10
```

### MultivariateConfig

Configuration for multivariate analysis.

```python
@dataclass
class MultivariateConfig:
    pca_components: Optional[int] = None
    pca_variance_threshold: float = 0.95
    clustering_method: str = "kmeans"
    n_clusters: Optional[int] = None
    scaling_method: str = "standard"
```

## Result Classes

### SensitivityResult

Result of sensitivity analysis.

```python
@dataclass
class SensitivityResult:
    success: bool
    method: str
    parameter_names: List[str]
    sensitivity_indices: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    analysis_time: float
    sample_size: int
    error_message: Optional[str] = None
```

### MultivariateResult

Result of multivariate analysis.

```python
@dataclass
class MultivariateResult:
    success: bool
    method: str
    feature_names: List[str]
    analysis_results: Dict[str, Any]
    explained_variance: Dict[str, float]
    component_loadings: Optional[pd.DataFrame] = None
    cluster_labels: Optional[np.ndarray] = None
    analysis_time: float = 0.0
    error_message: Optional[str] = None
```

## Error Handling

All analysis methods return result objects with success flags and error messages:

```python
result = analyzer.analyze_sobol(model_function, parameter_bounds)

if result.success:
    print("Analysis completed successfully")
    print(f"Analysis time: {{result.analysis_time:.2f}}s")
else:
    print(f"Analysis failed: {{result.error_message}}")
```

## Performance Tips

1. **Use Morris screening for initial parameter screening**
2. **Reduce sample size for faster analysis during development**
3. **Use parallel processing for large datasets**
4. **Cache results to avoid recomputation**

## Dependencies

- numpy >= 1.19.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0
- SALib >= 1.4.0 (for sensitivity analysis)
- pymc3 >= 3.11.0 (for Bayesian analysis, optional)
"""
        return markdown_content
    
    def _generate_api_documentation_html(self, doc_title: str) -> str:
        """Generate API documentation in HTML format."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{doc_title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; }}
        h3 {{ color: #7f8c8d; }}
        code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
        pre {{ background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        .class-doc {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>{doc_title}</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <h2>API Reference</h2>
    
    <div class="class-doc">
        <h3>GlobalSensitivityAnalyzer</h3>
        <p>Main class for global sensitivity analysis.</p>
        <pre><code>class GlobalSensitivityAnalyzer:
    def __init__(self, config: SensitivityConfig = None):
        \"\"\"Initialize the global sensitivity analyzer.\"\"\"
    
    def analyze_sobol(self, model_function, parameter_bounds, parameter_names=None):
        \"\"\"Perform Sobol sensitivity analysis.\"\"\"
    
    def analyze_morris(self, model_function, parameter_bounds, parameter_names=None):
        \"\"\"Perform Morris screening analysis.\"\"\"</code></pre>
    </div>
    
    <div class="class-doc">
        <h3>MultivariateAnalyzer</h3>
        <p>Main class for multivariate analysis.</p>
        <pre><code>class MultivariateAnalyzer:
    def __init__(self, config: MultivariateConfig = None):
        \"\"\"Initialize the multivariate analyzer.\"\"\"
    
    def analyze_pca(self, data, feature_names=None, n_components=None):
        \"\"\"Perform principal component analysis.\"\"\"
    
    def analyze_clustering(self, data, feature_names=None, method=None, n_clusters=None):
        \"\"\"Perform clustering analysis.\"\"\"</code></pre>
    </div>
    
    <div class="class-doc">
        <h3>ParameterAnalyzer</h3>
        <p>Main class for parameter analysis.</p>
        <pre><code>class ParameterAnalyzer:
    def __init__(self, config: ParameterAnalysisConfig = None):
        \"\"\"Initialize the parameter analyzer.\"\"\"
    
    def analyze_parameter_optimization(self, objective_function, parameter_bounds, parameter_names=None):
        \"\"\"Perform parameter optimization analysis.\"\"\"
    
    def analyze_parameter_interactions(self, process_data, parameter_names=None, target_variable=None):
        \"\"\"Analyze parameter interactions in process data.\"\"\"</code></pre>
    </div>
    
    <h2>Configuration Classes</h2>
    <p>Configuration classes define parameters for different analysis types.</p>
    
    <h2>Result Classes</h2>
    <p>Result classes contain the output of analysis methods.</p>
    
    <h2>Error Handling</h2>
    <p>All analysis methods return result objects with success flags and error messages.</p>
    
    <h2>Performance Tips</h2>
    <ul>
        <li>Use Morris screening for initial parameter screening</li>
        <li>Reduce sample size for faster analysis during development</li>
        <li>Use parallel processing for large datasets</li>
        <li>Cache results to avoid recomputation</li>
    </ul>
</body>
</html>
"""
        return html_content


class APIDocumentation(AnalysisDocumentation):
    """Specialized API documentation generator."""
    
    def __init__(self, config: DocumentationConfig = None):
        super().__init__(config)
        self.doc_type = "APIDocumentation"
    
    def generate_docs(self, doc_title: str = "PBF-LB/M Analytics API Documentation") -> DocumentationResult:
        """Generate API documentation."""
        return self.generate_api_documentation(doc_title)
