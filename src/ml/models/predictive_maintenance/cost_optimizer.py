"""
Cost Optimizer Model

This module implements a model for optimizing maintenance costs
and resource allocation for PBF-LB/M equipment maintenance.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import minimize
import time

from ..base_model import BaseModel

logger = logging.getLogger(__name__)


class CostOptimizer(BaseModel):
    """
    Model for optimizing maintenance costs and resource allocation.
    
    This model optimizes:
    - Maintenance cost minimization
    - Resource allocation efficiency
    - Downtime cost optimization
    - Spare parts inventory management
    - Labor cost optimization
    - Total cost of ownership (TCO)
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the cost optimizer.
        
        Args:
            config_manager: Configuration manager instance
        """
        super().__init__('cost_optimizer', config_manager)
        self.model_type = self.model_info.get('algorithm', 'neural_network')
        self.cost_components = {
            'labor': 0.4,      # 40% of total cost
            'parts': 0.3,      # 30% of total cost
            'downtime': 0.2,   # 20% of total cost
            'overhead': 0.1    # 10% of total cost
        }
        
        logger.info(f"Initialized CostOptimizer with algorithm: {self.model_type}")
    
    def build_model(self) -> Any:
        """
        Build the model architecture based on configuration.
        
        Returns:
            Built model instance
        """
        try:
            arch_config = self.architecture
            algorithm = arch_config.get('algorithm', 'neural_network')
            
            if algorithm == 'neural_network':
                return self._build_neural_network()
            elif algorithm == 'random_forest':
                return self._build_random_forest()
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
                
        except Exception as e:
            logger.error(f"Failed to build model: {e}")
            raise
    
    def _build_neural_network(self) -> tf.keras.Model:
        """Build neural network model."""
        model = Sequential()
        
        # Input layer
        input_dim = len(self.get_feature_names())
        model.add(Dense(256, activation='relu', input_dim=input_dim))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        # Hidden layers
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.1))
        
        # Output layer (total cost prediction)
        model.add(Dense(1, activation='linear'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.training_config.get('learning_rate', 0.001)),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def _build_random_forest(self) -> RandomForestRegressor:
        """Build random forest model."""
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training features (maintenance parameters, resource usage, etc.)
            y_train: Training targets (maintenance costs)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Training history dictionary
        """
        try:
            # Build model if not already built
            if self.model is None:
                self.model = self.build_model()
            
            start_time = time.time()
            
            if self.model_type == 'neural_network':
                history = self._train_neural_network(X_train, y_train, X_val, y_val)
            else:
                history = self._train_random_forest(X_train, y_train, X_val, y_val)
            
            training_time = time.time() - start_time
            
            self.training_history = {
                'training_time': training_time,
                'history': history
            }
            
            self.is_trained = True
            
            logger.info(f"Model training completed in {training_time:.2f} seconds")
            
            return self.training_history
            
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            raise
    
    def _train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray, 
                             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train neural network model."""
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.training_config.get('early_stopping', {}).get('patience', 15),
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.training_config.get('batch_size', 32),
            epochs=self.training_config.get('epochs', 150),
            validation_data=(X_val, y_val) if X_val is not None else None,
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history
    
    def _train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, 
                            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train random forest model."""
        self.model.fit(X_train, y_train)
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        train_mse = mean_squared_error(y_train, train_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)
        
        history = {
            'loss': [train_mse],
            'mae': [train_mae],
            'r2': [train_r2]
        }
        
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_mse = mean_squared_error(y_val, val_pred)
            val_mae = mean_absolute_error(y_val, val_pred)
            val_r2 = r2_score(y_val, val_pred)
            
            history.update({
                'val_loss': [val_mse],
                'val_mae': [val_mae],
                'val_r2': [val_r2]
            })
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            
        Returns:
            Predictions array (maintenance costs)
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            predictions = self.model.predict(X, verbose=0)
            
            if self.model_type == 'neural_network':
                return predictions.flatten()
            else:
                return predictions
            
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            raise
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets (maintenance costs)
            
        Returns:
            Evaluation metrics dictionary
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before evaluation")
            
            # Make predictions
            predictions = self.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            # Calculate percentage error
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
            
            # Calculate latency
            start_time = time.time()
            _ = self.predict(X_test[:10])
            latency = (time.time() - start_time) / 10 * 1000
            
            self.evaluation_metrics = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'latency_ms': latency,
                'test_samples': len(X_test)
            }
            
            logger.info(f"Model evaluation completed:")
            logger.info(f"  MSE: {mse:.4f}")
            logger.info(f"  MAE: {mae:.4f}")
            logger.info(f"  R²: {r2:.4f}")
            logger.info(f"  MAPE: {mape:.2f}%")
            logger.info(f"  Latency: {latency:.2f} ms")
            
            return self.evaluation_metrics
            
        except Exception as e:
            logger.error(f"Failed to evaluate model: {e}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance.
        
        Returns:
            Feature importance dictionary
        """
        if self.model_type == 'random_forest' and hasattr(self.model, 'feature_importances_'):
            feature_names = self.get_feature_names()
            importances = self.model.feature_importances_
            return dict(zip(feature_names, importances))
        else:
            # For neural networks, return zero importance
            feature_names = self.get_feature_names()
            return {name: 0.0 for name in feature_names}
    
    def optimize_maintenance_costs(self, maintenance_requirements: List[Dict[str, Any]], 
                                 resource_constraints: Dict[str, Any],
                                 cost_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize maintenance costs for given requirements and constraints.
        
        Args:
            maintenance_requirements: List of maintenance requirements
            resource_constraints: Resource availability constraints
            cost_parameters: Cost parameters and rates
            
        Returns:
            Dictionary with cost optimization results
        """
        try:
            # Analyze cost components
            cost_breakdown = self._analyze_cost_components(maintenance_requirements, cost_parameters)
            
            # Optimize resource allocation
            optimized_allocation = self._optimize_resource_allocation(
                maintenance_requirements, resource_constraints, cost_parameters
            )
            
            # Calculate cost savings
            cost_savings = self._calculate_cost_savings(cost_breakdown, optimized_allocation)
            
            # Generate optimization recommendations
            recommendations = self._generate_cost_optimization_recommendations(
                cost_breakdown, optimized_allocation, cost_savings
            )
            
            return {
                'cost_breakdown': cost_breakdown,
                'optimized_allocation': optimized_allocation,
                'cost_savings': cost_savings,
                'recommendations': recommendations,
                'total_cost': cost_breakdown['total_cost'],
                'optimized_cost': optimized_allocation['total_cost'],
                'savings_percentage': cost_savings['percentage_savings']
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize maintenance costs: {e}")
            raise
    
    def _analyze_cost_components(self, maintenance_requirements: List[Dict[str, Any]], 
                               cost_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze cost components for maintenance requirements.
        
        Args:
            maintenance_requirements: List of maintenance requirements
            cost_parameters: Cost parameters and rates
            
        Returns:
            Dictionary with cost breakdown
        """
        total_labor_cost = 0
        total_parts_cost = 0
        total_downtime_cost = 0
        total_overhead_cost = 0
        
        for requirement in maintenance_requirements:
            equipment_id = requirement['equipment_id']
            maintenance_type = requirement['maintenance_type']
            duration_hours = requirement['estimated_duration_hours']
            
            # Calculate labor cost
            labor_rate = cost_parameters.get('labor_rate', 50)  # $/hour
            labor_cost = duration_hours * labor_rate
            total_labor_cost += labor_cost
            
            # Calculate parts cost
            parts_cost = self._calculate_parts_cost(equipment_id, maintenance_type, cost_parameters)
            total_parts_cost += parts_cost
            
            # Calculate downtime cost
            downtime_rate = cost_parameters.get('downtime_rate', 1000)  # $/hour
            downtime_cost = duration_hours * downtime_rate
            total_downtime_cost += downtime_cost
            
            # Calculate overhead cost
            overhead_rate = cost_parameters.get('overhead_rate', 0.1)  # 10% of total
            overhead_cost = (labor_cost + parts_cost) * overhead_rate
            total_overhead_cost += overhead_cost
        
        total_cost = total_labor_cost + total_parts_cost + total_downtime_cost + total_overhead_cost
        
        return {
            'labor_cost': total_labor_cost,
            'parts_cost': total_parts_cost,
            'downtime_cost': total_downtime_cost,
            'overhead_cost': total_overhead_cost,
            'total_cost': total_cost,
            'cost_distribution': {
                'labor_percentage': (total_labor_cost / total_cost) * 100,
                'parts_percentage': (total_parts_cost / total_cost) * 100,
                'downtime_percentage': (total_downtime_cost / total_cost) * 100,
                'overhead_percentage': (total_overhead_cost / total_cost) * 100
            }
        }
    
    def _calculate_parts_cost(self, equipment_id: str, maintenance_type: str, 
                            cost_parameters: Dict[str, Any]) -> float:
        """
        Calculate parts cost for maintenance.
        
        Args:
            equipment_id: Equipment identifier
            maintenance_type: Type of maintenance
            cost_parameters: Cost parameters
            
        Returns:
            Parts cost
        """
        # Base parts costs by equipment type
        base_parts_costs = {
            'laser': {'preventive': 500, 'predictive': 200, 'corrective': 2000, 'emergency': 3000},
            'recoater': {'preventive': 300, 'predictive': 100, 'corrective': 1500, 'emergency': 2500},
            'chamber': {'preventive': 800, 'predictive': 300, 'corrective': 3000, 'emergency': 5000},
            'gas_system': {'preventive': 400, 'predictive': 150, 'corrective': 1800, 'emergency': 2800}
        }
        
        equipment_type = equipment_id.split('_')[0]
        base_cost = base_parts_costs.get(equipment_type, {}).get(maintenance_type, 1000)
        
        # Apply cost multiplier if available
        cost_multiplier = cost_parameters.get('parts_cost_multiplier', 1.0)
        
        return base_cost * cost_multiplier
    
    def _optimize_resource_allocation(self, maintenance_requirements: List[Dict[str, Any]], 
                                    resource_constraints: Dict[str, Any],
                                    cost_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize resource allocation to minimize costs.
        
        Args:
            maintenance_requirements: List of maintenance requirements
            resource_constraints: Resource availability constraints
            cost_parameters: Cost parameters
            
        Returns:
            Dictionary with optimized allocation
        """
        # Sort requirements by cost-benefit ratio
        prioritized_requirements = self._prioritize_requirements(maintenance_requirements, cost_parameters)
        
        # Allocate resources optimally
        allocated_resources = {}
        total_optimized_cost = 0
        
        for requirement in prioritized_requirements:
            equipment_id = requirement['equipment_id']
            
            # Find optimal resource allocation
            optimal_allocation = self._find_optimal_allocation(
                requirement, resource_constraints, cost_parameters
            )
            
            if optimal_allocation:
                allocated_resources[equipment_id] = optimal_allocation
                total_optimized_cost += optimal_allocation['total_cost']
        
        return {
            'allocated_resources': allocated_resources,
            'total_cost': total_optimized_cost,
            'resource_utilization': self._calculate_resource_utilization(allocated_resources, resource_constraints)
        }
    
    def _prioritize_requirements(self, maintenance_requirements: List[Dict[str, Any]], 
                               cost_parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Prioritize maintenance requirements based on cost-benefit analysis.
        
        Args:
            maintenance_requirements: List of maintenance requirements
            cost_parameters: Cost parameters
            
        Returns:
            Prioritized list of requirements
        """
        prioritized = []
        
        for requirement in maintenance_requirements:
            # Calculate cost-benefit ratio
            cost = self._estimate_maintenance_cost(requirement, cost_parameters)
            benefit = self._estimate_maintenance_benefit(requirement)
            
            cost_benefit_ratio = benefit / cost if cost > 0 else 0
            
            prioritized.append({
                **requirement,
                'estimated_cost': cost,
                'estimated_benefit': benefit,
                'cost_benefit_ratio': cost_benefit_ratio
            })
        
        # Sort by cost-benefit ratio (descending)
        prioritized.sort(key=lambda x: x['cost_benefit_ratio'], reverse=True)
        
        return prioritized
    
    def _estimate_maintenance_cost(self, requirement: Dict[str, Any], 
                                 cost_parameters: Dict[str, Any]) -> float:
        """
        Estimate maintenance cost for a requirement.
        
        Args:
            requirement: Maintenance requirement
            cost_parameters: Cost parameters
            
        Returns:
            Estimated cost
        """
        duration_hours = requirement['estimated_duration_hours']
        maintenance_type = requirement['maintenance_type']
        equipment_id = requirement['equipment_id']
        
        # Labor cost
        labor_rate = cost_parameters.get('labor_rate', 50)
        labor_cost = duration_hours * labor_rate
        
        # Parts cost
        parts_cost = self._calculate_parts_cost(equipment_id, maintenance_type, cost_parameters)
        
        # Downtime cost
        downtime_rate = cost_parameters.get('downtime_rate', 1000)
        downtime_cost = duration_hours * downtime_rate
        
        # Overhead cost
        overhead_rate = cost_parameters.get('overhead_rate', 0.1)
        overhead_cost = (labor_cost + parts_cost) * overhead_rate
        
        return labor_cost + parts_cost + downtime_cost + overhead_cost
    
    def _estimate_maintenance_benefit(self, requirement: Dict[str, Any]) -> float:
        """
        Estimate maintenance benefit (cost avoidance).
        
        Args:
            requirement: Maintenance requirement
            
        Returns:
            Estimated benefit
        """
        # Benefit is based on failure probability and potential failure cost
        failure_probability = requirement.get('failure_probability', 0.0)
        potential_failure_cost = requirement.get('potential_failure_cost', 10000)
        
        # Benefit = failure_probability * potential_failure_cost
        benefit = failure_probability * potential_failure_cost
        
        # Add benefit for preventive maintenance
        if requirement['maintenance_type'] == 'preventive':
            benefit *= 1.5  # Preventive maintenance has higher benefit
        
        return benefit
    
    def _find_optimal_allocation(self, requirement: Dict[str, Any], 
                               resource_constraints: Dict[str, Any],
                               cost_parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Find optimal resource allocation for a requirement.
        
        Args:
            requirement: Maintenance requirement
            resource_constraints: Resource constraints
            cost_parameters: Cost parameters
            
        Returns:
            Optimal allocation or None if not feasible
        """
        # Check if resources are available
        if not self._check_resource_availability(requirement, resource_constraints):
            return None
        
        # Calculate optimal allocation cost
        duration_hours = requirement['estimated_duration_hours']
        maintenance_type = requirement['maintenance_type']
        equipment_id = requirement['equipment_id']
        
        # Labor cost
        labor_rate = cost_parameters.get('labor_rate', 50)
        labor_cost = duration_hours * labor_rate
        
        # Parts cost
        parts_cost = self._calculate_parts_cost(equipment_id, maintenance_type, cost_parameters)
        
        # Downtime cost (can be optimized by scheduling)
        downtime_rate = cost_parameters.get('downtime_rate', 1000)
        downtime_cost = duration_hours * downtime_rate
        
        # Overhead cost
        overhead_rate = cost_parameters.get('overhead_rate', 0.1)
        overhead_cost = (labor_cost + parts_cost) * overhead_rate
        
        total_cost = labor_cost + parts_cost + downtime_cost + overhead_cost
        
        return {
            'labor_cost': labor_cost,
            'parts_cost': parts_cost,
            'downtime_cost': downtime_cost,
            'overhead_cost': overhead_cost,
            'total_cost': total_cost,
            'allocated_resources': {
                'technician_hours': duration_hours,
                'parts_required': maintenance_type in ['corrective', 'emergency'],
                'tools_required': True
            }
        }
    
    def _check_resource_availability(self, requirement: Dict[str, Any], 
                                   resource_constraints: Dict[str, Any]) -> bool:
        """
        Check if resources are available for a requirement.
        
        Args:
            requirement: Maintenance requirement
            resource_constraints: Resource constraints
            
        Returns:
            True if resources are available
        """
        # Check technician availability
        if 'technician' not in resource_constraints or resource_constraints['technician']['count'] <= 0:
            return False
        
        # Check tools availability
        if 'tools' not in resource_constraints or resource_constraints['tools']['count'] <= 0:
            return False
        
        # Check parts availability for corrective/emergency maintenance
        if requirement['maintenance_type'] in ['corrective', 'emergency']:
            if 'parts' not in resource_constraints or resource_constraints['parts']['count'] <= 0:
                return False
        
        return True
    
    def _calculate_resource_utilization(self, allocated_resources: Dict[str, Any], 
                                      resource_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate resource utilization.
        
        Args:
            allocated_resources: Allocated resources
            resource_constraints: Resource constraints
            
        Returns:
            Resource utilization metrics
        """
        utilization = {}
        
        for resource_type, resource_info in resource_constraints.items():
            total_available = resource_info.get('count', 0)
            used_count = len(allocated_resources)  # Simplified
            
            utilization[resource_type] = {
                'total_available': total_available,
                'used': used_count,
                'utilization_rate': used_count / total_available if total_available > 0 else 0.0,
                'efficiency': min(1.0, used_count / total_available) if total_available > 0 else 0.0
            }
        
        return utilization
    
    def _calculate_cost_savings(self, cost_breakdown: Dict[str, Any], 
                              optimized_allocation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate cost savings from optimization.
        
        Args:
            cost_breakdown: Original cost breakdown
            optimized_allocation: Optimized allocation
            
        Returns:
            Cost savings metrics
        """
        original_cost = cost_breakdown['total_cost']
        optimized_cost = optimized_allocation['total_cost']
        
        absolute_savings = original_cost - optimized_cost
        percentage_savings = (absolute_savings / original_cost) * 100 if original_cost > 0 else 0
        
        return {
            'absolute_savings': absolute_savings,
            'percentage_savings': percentage_savings,
            'original_cost': original_cost,
            'optimized_cost': optimized_cost,
            'savings_breakdown': {
                'labor_savings': cost_breakdown['labor_cost'] - optimized_allocation.get('labor_cost', 0),
                'parts_savings': cost_breakdown['parts_cost'] - optimized_allocation.get('parts_cost', 0),
                'downtime_savings': cost_breakdown['downtime_cost'] - optimized_allocation.get('downtime_cost', 0)
            }
        }
    
    def _generate_cost_optimization_recommendations(self, cost_breakdown: Dict[str, Any], 
                                                  optimized_allocation: Dict[str, Any],
                                                  cost_savings: Dict[str, Any]) -> List[str]:
        """
        Generate cost optimization recommendations.
        
        Args:
            cost_breakdown: Original cost breakdown
            optimized_allocation: Optimized allocation
            cost_savings: Cost savings metrics
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Overall savings recommendation
        if cost_savings['percentage_savings'] > 10:
            recommendations.append(f"Excellent optimization achieved: {cost_savings['percentage_savings']:.1f}% cost reduction")
        elif cost_savings['percentage_savings'] > 5:
            recommendations.append(f"Good optimization achieved: {cost_savings['percentage_savings']:.1f}% cost reduction")
        else:
            recommendations.append(f"Modest optimization achieved: {cost_savings['percentage_savings']:.1f}% cost reduction")
        
        # Component-specific recommendations
        if cost_breakdown['downtime_cost'] > cost_breakdown['labor_cost']:
            recommendations.append("Downtime cost is highest - focus on reducing maintenance duration")
        
        if cost_breakdown['parts_cost'] > cost_breakdown['labor_cost']:
            recommendations.append("Parts cost is high - consider bulk purchasing or alternative suppliers")
        
        if cost_breakdown['labor_cost'] > cost_breakdown['parts_cost']:
            recommendations.append("Labor cost is high - consider automation or skill optimization")
        
        # Resource utilization recommendations
        resource_utilization = optimized_allocation.get('resource_utilization', {})
        for resource_type, utilization in resource_utilization.items():
            if utilization['utilization_rate'] < 0.7:
                recommendations.append(f"Low {resource_type} utilization - consider resource sharing or scheduling optimization")
        
        return recommendations
