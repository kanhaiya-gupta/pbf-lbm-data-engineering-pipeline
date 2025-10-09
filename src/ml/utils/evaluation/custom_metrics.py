"""
Custom Metrics

This module implements custom evaluation metrics specific to
PBF-LB/M manufacturing processes.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class CustomMetrics:
    """
    Utility class for custom evaluation metrics specific to manufacturing.
    
    This class handles:
    - Manufacturing-specific quality metrics
    - Process optimization metrics
    - Cost-effectiveness metrics
    - Production efficiency metrics
    - Quality control metrics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the custom metrics calculator.
        
        Args:
            config: Configuration dictionary with metric settings
        """
        self.config = config or {}
        
        # Manufacturing-specific metrics
        self.manufacturing_metrics = {
            'dimensional_accuracy': self._calculate_dimensional_accuracy,
            'surface_roughness_accuracy': self._calculate_surface_roughness_accuracy,
            'mechanical_property_accuracy': self._calculate_mechanical_property_accuracy,
            'build_time_accuracy': self._calculate_build_time_accuracy,
            'material_usage_accuracy': self._calculate_material_usage_accuracy,
            'energy_consumption_accuracy': self._calculate_energy_consumption_accuracy,
            'defect_detection_accuracy': self._calculate_defect_detection_accuracy,
            'process_parameter_optimization': self._calculate_process_parameter_optimization,
            'quality_control_score': self._calculate_quality_control_score,
            'production_efficiency': self._calculate_production_efficiency,
            'cost_effectiveness': self._calculate_cost_effectiveness,
            'sustainability_score': self._calculate_sustainability_score
        }
        
        # Default tolerances for manufacturing metrics
        self.default_tolerances = {
            'dimensional_accuracy': 0.01,  # 0.01mm
            'surface_roughness': 0.5,      # 0.5 micrometers
            'mechanical_properties': 0.05,  # 5%
            'build_time': 0.1,             # 10%
            'material_usage': 0.05,        # 5%
            'energy_consumption': 0.1      # 10%
        }
        
        logger.info("Initialized CustomMetrics")
    
    def calculate_metrics(self, y_true: Union[np.ndarray, pd.Series, List], 
                         y_pred: Union[np.ndarray, pd.Series, List],
                         metric_type: str,
                         specifications: Optional[Dict[str, Any]] = None,
                         cost_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Calculate custom manufacturing metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            metric_type: Type of metric to calculate
            specifications: Manufacturing specifications and tolerances
            cost_data: Cost data for cost-effectiveness calculations
            
        Returns:
            Dictionary with calculated metrics
        """
        # Convert inputs to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Validate inputs
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        if len(y_true) == 0:
            raise ValueError("Input arrays cannot be empty")
        
        # Remove any NaN values
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        
        if len(y_true) == 0:
            raise ValueError("No valid values found after removing NaN")
        
        results = {}
        
        if metric_type in self.manufacturing_metrics:
            try:
                results[metric_type] = self.manufacturing_metrics[metric_type](
                    y_true, y_pred, specifications, cost_data
                )
            except Exception as e:
                logger.error(f"Failed to calculate {metric_type}: {e}")
                results[metric_type] = np.nan
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")
        
        return results
    
    def _calculate_dimensional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      specifications: Optional[Dict[str, Any]] = None,
                                      cost_data: Optional[Dict[str, Any]] = None) -> float:
        """Calculate dimensional accuracy for manufacturing."""
        if specifications is None:
            specifications = {}
        
        tolerance = specifications.get('tolerance', self.default_tolerances['dimensional_accuracy'])
        tolerance_type = specifications.get('tolerance_type', 'absolute')
        
        if tolerance_type == 'absolute':
            errors = np.abs(y_true - y_pred)
            within_tolerance = errors <= tolerance
        elif tolerance_type == 'relative':
            errors = np.abs(y_true - y_pred) / np.abs(y_true)
            within_tolerance = errors <= tolerance
        else:
            raise ValueError(f"Unknown tolerance type: {tolerance_type}")
        
        accuracy = np.mean(within_tolerance) * 100
        return float(accuracy)
    
    def _calculate_surface_roughness_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                            specifications: Optional[Dict[str, Any]] = None,
                                            cost_data: Optional[Dict[str, Any]] = None) -> float:
        """Calculate surface roughness prediction accuracy."""
        if specifications is None:
            specifications = {}
        
        tolerance = specifications.get('tolerance', self.default_tolerances['surface_roughness'])
        tolerance_type = specifications.get('tolerance_type', 'absolute')
        
        if tolerance_type == 'absolute':
            errors = np.abs(y_true - y_pred)
            within_tolerance = errors <= tolerance
        elif tolerance_type == 'relative':
            errors = np.abs(y_true - y_pred) / np.abs(y_true)
            within_tolerance = errors <= tolerance
        else:
            raise ValueError(f"Unknown tolerance type: {tolerance_type}")
        
        accuracy = np.mean(within_tolerance) * 100
        return float(accuracy)
    
    def _calculate_mechanical_property_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                              specifications: Optional[Dict[str, Any]] = None,
                                              cost_data: Optional[Dict[str, Any]] = None) -> float:
        """Calculate mechanical property prediction accuracy."""
        if specifications is None:
            specifications = {}
        
        tolerance = specifications.get('tolerance', self.default_tolerances['mechanical_properties'])
        tolerance_type = specifications.get('tolerance_type', 'relative')
        
        if tolerance_type == 'absolute':
            errors = np.abs(y_true - y_pred)
            within_tolerance = errors <= tolerance
        elif tolerance_type == 'relative':
            errors = np.abs(y_true - y_pred) / np.abs(y_true)
            within_tolerance = errors <= tolerance
        else:
            raise ValueError(f"Unknown tolerance type: {tolerance_type}")
        
        accuracy = np.mean(within_tolerance) * 100
        return float(accuracy)
    
    def _calculate_build_time_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                     specifications: Optional[Dict[str, Any]] = None,
                                     cost_data: Optional[Dict[str, Any]] = None) -> float:
        """Calculate build time prediction accuracy."""
        if specifications is None:
            specifications = {}
        
        tolerance = specifications.get('tolerance', self.default_tolerances['build_time'])
        tolerance_type = specifications.get('tolerance_type', 'relative')
        
        if tolerance_type == 'absolute':
            errors = np.abs(y_true - y_pred)
            within_tolerance = errors <= tolerance
        elif tolerance_type == 'relative':
            errors = np.abs(y_true - y_pred) / np.abs(y_true)
            within_tolerance = errors <= tolerance
        else:
            raise ValueError(f"Unknown tolerance type: {tolerance_type}")
        
        accuracy = np.mean(within_tolerance) * 100
        return float(accuracy)
    
    def _calculate_material_usage_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                         specifications: Optional[Dict[str, Any]] = None,
                                         cost_data: Optional[Dict[str, Any]] = None) -> float:
        """Calculate material usage prediction accuracy."""
        if specifications is None:
            specifications = {}
        
        tolerance = specifications.get('tolerance', self.default_tolerances['material_usage'])
        tolerance_type = specifications.get('tolerance_type', 'relative')
        
        if tolerance_type == 'absolute':
            errors = np.abs(y_true - y_pred)
            within_tolerance = errors <= tolerance
        elif tolerance_type == 'relative':
            errors = np.abs(y_true - y_pred) / np.abs(y_true)
            within_tolerance = errors <= tolerance
        else:
            raise ValueError(f"Unknown tolerance type: {tolerance_type}")
        
        accuracy = np.mean(within_tolerance) * 100
        return float(accuracy)
    
    def _calculate_energy_consumption_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                             specifications: Optional[Dict[str, Any]] = None,
                                             cost_data: Optional[Dict[str, Any]] = None) -> float:
        """Calculate energy consumption prediction accuracy."""
        if specifications is None:
            specifications = {}
        
        tolerance = specifications.get('tolerance', self.default_tolerances['energy_consumption'])
        tolerance_type = specifications.get('tolerance_type', 'relative')
        
        if tolerance_type == 'absolute':
            errors = np.abs(y_true - y_pred)
            within_tolerance = errors <= tolerance
        elif tolerance_type == 'relative':
            errors = np.abs(y_true - y_pred) / np.abs(y_true)
            within_tolerance = errors <= tolerance
        else:
            raise ValueError(f"Unknown tolerance type: {tolerance_type}")
        
        accuracy = np.mean(within_tolerance) * 100
        return float(accuracy)
    
    def _calculate_defect_detection_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                           specifications: Optional[Dict[str, Any]] = None,
                                           cost_data: Optional[Dict[str, Any]] = None) -> float:
        """Calculate defect detection accuracy."""
        # Assuming 1 = defect, 0 = no defect
        if len(np.unique(y_true)) == 2 and len(np.unique(y_pred)) == 2:
            # Binary classification
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(y_true, y_pred)
            return float(accuracy * 100)
        else:
            # Regression case - use threshold-based classification
            threshold = specifications.get('defect_threshold', 0.5) if specifications else 0.5
            
            y_true_binary = (y_true > threshold).astype(int)
            y_pred_binary = (y_pred > threshold).astype(int)
            
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(y_true_binary, y_pred_binary)
            return float(accuracy * 100)
    
    def _calculate_process_parameter_optimization(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                                specifications: Optional[Dict[str, Any]] = None,
                                                cost_data: Optional[Dict[str, Any]] = None) -> float:
        """Calculate process parameter optimization score."""
        if specifications is None:
            specifications = {}
        
        # Get target values and acceptable ranges
        target_values = specifications.get('target_values', np.mean(y_true))
        acceptable_ranges = specifications.get('acceptable_ranges', np.std(y_true))
        
        if isinstance(target_values, (int, float)):
            target_values = np.full_like(y_true, target_values)
        if isinstance(acceptable_ranges, (int, float)):
            acceptable_ranges = np.full_like(y_true, acceptable_ranges)
        
        # Calculate optimization score
        optimization_scores = []
        for i in range(len(y_true)):
            target = target_values[i]
            range_val = acceptable_ranges[i]
            
            # Calculate how close prediction is to target
            distance_from_target = abs(y_pred[i] - target)
            
            # Normalize by acceptable range
            if range_val > 0:
                normalized_distance = distance_from_target / range_val
                # Convert to score (0-100)
                score = max(0, 100 - normalized_distance * 100)
            else:
                score = 100 if distance_from_target == 0 else 0
            
            optimization_scores.append(score)
        
        return float(np.mean(optimization_scores))
    
    def _calculate_quality_control_score(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                       specifications: Optional[Dict[str, Any]] = None,
                                       cost_data: Optional[Dict[str, Any]] = None) -> float:
        """Calculate overall quality control score."""
        # Combine multiple quality metrics
        dimensional_accuracy = self._calculate_dimensional_accuracy(y_true, y_pred, specifications)
        surface_roughness_accuracy = self._calculate_surface_roughness_accuracy(y_true, y_pred, specifications)
        mechanical_property_accuracy = self._calculate_mechanical_property_accuracy(y_true, y_pred, specifications)
        
        # Weighted combination
        quality_score = (
            dimensional_accuracy * 0.4 +
            surface_roughness_accuracy * 0.3 +
            mechanical_property_accuracy * 0.3
        )
        
        return float(quality_score)
    
    def _calculate_production_efficiency(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                       specifications: Optional[Dict[str, Any]] = None,
                                       cost_data: Optional[Dict[str, Any]] = None) -> float:
        """Calculate production efficiency score."""
        # Production efficiency based on accuracy and consistency
        accuracy = self._calculate_dimensional_accuracy(y_true, y_pred, specifications)
        
        # Calculate consistency (inverse of coefficient of variation)
        errors = np.abs(y_true - y_pred)
        if np.mean(errors) > 0:
            consistency = 100 - (np.std(errors) / np.mean(errors)) * 100
        else:
            consistency = 100
        
        # Combine accuracy and consistency
        efficiency = (accuracy * 0.6 + consistency * 0.4)
        
        return float(max(0, min(100, efficiency)))
    
    def _calculate_cost_effectiveness(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    specifications: Optional[Dict[str, Any]] = None,
                                    cost_data: Optional[Dict[str, Any]] = None) -> float:
        """Calculate cost-effectiveness score."""
        if cost_data is None:
            return np.nan
        
        # Get cost parameters
        material_cost = cost_data.get('material_cost', 1.0)
        energy_cost = cost_data.get('energy_cost', 1.0)
        labor_cost = cost_data.get('labor_cost', 1.0)
        rework_cost = cost_data.get('rework_cost', 10.0)
        
        # Calculate prediction accuracy
        accuracy = self._calculate_dimensional_accuracy(y_true, y_pred, specifications)
        
        # Calculate cost savings from accurate predictions
        # Accurate predictions reduce rework costs
        rework_reduction = (accuracy / 100) * rework_cost
        
        # Calculate total cost
        total_cost = material_cost + energy_cost + labor_cost
        cost_savings = rework_reduction
        
        # Calculate cost-effectiveness
        if total_cost > 0:
            cost_effectiveness = (cost_savings / total_cost) * 100
        else:
            cost_effectiveness = 0
        
        return float(max(0, min(100, cost_effectiveness)))
    
    def _calculate_sustainability_score(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      specifications: Optional[Dict[str, Any]] = None,
                                      cost_data: Optional[Dict[str, Any]] = None) -> float:
        """Calculate sustainability score."""
        # Sustainability based on material usage and energy consumption accuracy
        material_accuracy = self._calculate_material_usage_accuracy(y_true, y_pred, specifications)
        energy_accuracy = self._calculate_energy_consumption_accuracy(y_true, y_pred, specifications)
        
        # Combine material and energy efficiency
        sustainability_score = (material_accuracy * 0.6 + energy_accuracy * 0.4)
        
        return float(sustainability_score)
    
    def calculate_comprehensive_manufacturing_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                                    specifications: Optional[Dict[str, Any]] = None,
                                                    cost_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Calculate comprehensive manufacturing metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            specifications: Manufacturing specifications
            cost_data: Cost data for calculations
            
        Returns:
            Dictionary with all manufacturing metrics
        """
        results = {}
        
        for metric_name, metric_func in self.manufacturing_metrics.items():
            try:
                results[metric_name] = metric_func(y_true, y_pred, specifications, cost_data)
            except Exception as e:
                logger.warning(f"Failed to calculate {metric_name}: {e}")
                results[metric_name] = np.nan
        
        return results
    
    def calculate_manufacturing_kpis(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   specifications: Optional[Dict[str, Any]] = None,
                                   cost_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Calculate manufacturing KPIs.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            specifications: Manufacturing specifications
            cost_data: Cost data for calculations
            
        Returns:
            Dictionary with manufacturing KPIs
        """
        # Calculate all metrics
        metrics = self.calculate_comprehensive_manufacturing_metrics(y_true, y_pred, specifications, cost_data)
        
        # Calculate KPIs
        kpis = {
            'quality_kpis': {
                'dimensional_accuracy': metrics.get('dimensional_accuracy', 0),
                'surface_roughness_accuracy': metrics.get('surface_roughness_accuracy', 0),
                'mechanical_property_accuracy': metrics.get('mechanical_property_accuracy', 0),
                'overall_quality_score': metrics.get('quality_control_score', 0)
            },
            'efficiency_kpis': {
                'build_time_accuracy': metrics.get('build_time_accuracy', 0),
                'material_usage_accuracy': metrics.get('material_usage_accuracy', 0),
                'energy_consumption_accuracy': metrics.get('energy_consumption_accuracy', 0),
                'production_efficiency': metrics.get('production_efficiency', 0)
            },
            'cost_kpis': {
                'cost_effectiveness': metrics.get('cost_effectiveness', 0),
                'sustainability_score': metrics.get('sustainability_score', 0)
            },
            'process_kpis': {
                'defect_detection_accuracy': metrics.get('defect_detection_accuracy', 0),
                'process_parameter_optimization': metrics.get('process_parameter_optimization', 0)
            }
        }
        
        # Calculate overall KPI score
        all_scores = [score for kpi_group in kpis.values() for score in kpi_group.values() if not np.isnan(score)]
        if all_scores:
            kpis['overall_kpi_score'] = float(np.mean(all_scores))
        else:
            kpis['overall_kpi_score'] = 0.0
        
        return kpis
    
    def visualize_manufacturing_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      specifications: Optional[Dict[str, Any]] = None,
                                      cost_data: Optional[Dict[str, Any]] = None,
                                      title: str = "Manufacturing Metrics",
                                      save_path: Optional[str] = None) -> None:
        """
        Visualize manufacturing metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            specifications: Manufacturing specifications
            cost_data: Cost data for calculations
            title: Plot title
            save_path: Path to save the plot
        """
        # Calculate KPIs
        kpis = self.calculate_manufacturing_kpis(y_true, y_pred, specifications, cost_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Quality KPIs
        quality_kpis = kpis['quality_kpis']
        quality_names = list(quality_kpis.keys())
        quality_values = list(quality_kpis.values())
        
        axes[0, 0].bar(quality_names, quality_values, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Quality KPIs')
        axes[0, 0].set_ylabel('Score (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylim(0, 100)
        
        # Efficiency KPIs
        efficiency_kpis = kpis['efficiency_kpis']
        efficiency_names = list(efficiency_kpis.keys())
        efficiency_values = list(efficiency_kpis.values())
        
        axes[0, 1].bar(efficiency_names, efficiency_values, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('Efficiency KPIs')
        axes[0, 1].set_ylabel('Score (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].set_ylim(0, 100)
        
        # Cost KPIs
        cost_kpis = kpis['cost_kpis']
        cost_names = list(cost_kpis.keys())
        cost_values = list(cost_kpis.values())
        
        axes[1, 0].bar(cost_names, cost_values, color='lightcoral', alpha=0.7)
        axes[1, 0].set_title('Cost KPIs')
        axes[1, 0].set_ylabel('Score (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].set_ylim(0, 100)
        
        # Process KPIs
        process_kpis = kpis['process_kpis']
        process_names = list(process_kpis.keys())
        process_values = list(process_kpis.values())
        
        axes[1, 1].bar(process_names, process_values, color='lightyellow', alpha=0.7)
        axes[1, 1].set_title('Process KPIs')
        axes[1, 1].set_ylabel('Score (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].set_ylim(0, 100)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Manufacturing metrics visualization saved to {save_path}")
        
        plt.show()
    
    def generate_manufacturing_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    specifications: Optional[Dict[str, Any]] = None,
                                    cost_data: Optional[Dict[str, Any]] = None,
                                    model_name: str = "Model") -> str:
        """
        Generate a comprehensive manufacturing evaluation report.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            specifications: Manufacturing specifications
            cost_data: Cost data for calculations
            model_name: Name of the model
            
        Returns:
            Formatted manufacturing evaluation report
        """
        # Calculate KPIs
        kpis = self.calculate_manufacturing_kpis(y_true, y_pred, specifications, cost_data)
        
        # Generate report
        report = []
        report.append("=" * 60)
        report.append(f"MANUFACTURING MODEL EVALUATION REPORT")
        report.append(f"Model: {model_name}")
        report.append("=" * 60)
        report.append("")
        
        # Quality KPIs
        report.append("QUALITY KPIs:")
        quality_kpis = kpis['quality_kpis']
        report.append(f"  Dimensional Accuracy: {quality_kpis['dimensional_accuracy']:.2f}%")
        report.append(f"  Surface Roughness Accuracy: {quality_kpis['surface_roughness_accuracy']:.2f}%")
        report.append(f"  Mechanical Property Accuracy: {quality_kpis['mechanical_property_accuracy']:.2f}%")
        report.append(f"  Overall Quality Score: {quality_kpis['overall_quality_score']:.2f}%")
        report.append("")
        
        # Efficiency KPIs
        report.append("EFFICIENCY KPIs:")
        efficiency_kpis = kpis['efficiency_kpis']
        report.append(f"  Build Time Accuracy: {efficiency_kpis['build_time_accuracy']:.2f}%")
        report.append(f"  Material Usage Accuracy: {efficiency_kpis['material_usage_accuracy']:.2f}%")
        report.append(f"  Energy Consumption Accuracy: {efficiency_kpis['energy_consumption_accuracy']:.2f}%")
        report.append(f"  Production Efficiency: {efficiency_kpis['production_efficiency']:.2f}%")
        report.append("")
        
        # Cost KPIs
        report.append("COST KPIs:")
        cost_kpis = kpis['cost_kpis']
        report.append(f"  Cost Effectiveness: {cost_kpis['cost_effectiveness']:.2f}%")
        report.append(f"  Sustainability Score: {cost_kpis['sustainability_score']:.2f}%")
        report.append("")
        
        # Process KPIs
        report.append("PROCESS KPIs:")
        process_kpis = kpis['process_kpis']
        report.append(f"  Defect Detection Accuracy: {process_kpis['defect_detection_accuracy']:.2f}%")
        report.append(f"  Process Parameter Optimization: {process_kpis['process_parameter_optimization']:.2f}%")
        report.append("")
        
        # Overall KPI Score
        report.append("OVERALL PERFORMANCE:")
        overall_score = kpis['overall_kpi_score']
        report.append(f"  Overall KPI Score: {overall_score:.2f}%")
        
        if overall_score > 90:
            report.append("  Performance Rating: EXCELLENT (>90%)")
        elif overall_score > 80:
            report.append("  Performance Rating: GOOD (80-90%)")
        elif overall_score > 70:
            report.append("  Performance Rating: FAIR (70-80%)")
        else:
            report.append("  Performance Rating: POOR (<70%)")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
