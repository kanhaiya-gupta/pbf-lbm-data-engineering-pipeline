"""
Maintenance Scheduler Model

This module implements a model for optimizing maintenance scheduling
for PBF-LB/M equipment based on health monitoring and failure predictions.
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
from datetime import datetime, timedelta

from ..base_model import BaseModel

logger = logging.getLogger(__name__)


class MaintenanceScheduler(BaseModel):
    """
    Model for optimizing maintenance scheduling for PBF-LB/M equipment.
    
    This model optimizes:
    - Maintenance timing based on equipment health
    - Resource allocation for maintenance tasks
    - Maintenance type selection (preventive, corrective, predictive)
    - Downtime minimization
    - Cost optimization
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the maintenance scheduler.
        
        Args:
            config_manager: Configuration manager instance
        """
        super().__init__('maintenance_scheduler', config_manager)
        self.model_type = self.model_info.get('algorithm', 'neural_network')
        self.maintenance_types = ['preventive', 'corrective', 'predictive', 'emergency']
        
        logger.info(f"Initialized MaintenanceScheduler with algorithm: {self.model_type}")
    
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
        
        # Output layer (maintenance priority score)
        model.add(Dense(1, activation='sigmoid'))
        
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
            X_train: Training features (equipment health, usage, etc.)
            y_train: Training targets (maintenance priority scores)
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
            Predictions array (maintenance priority scores)
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
            y_test: Test targets (maintenance priority scores)
            
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
            
            # Calculate latency
            start_time = time.time()
            _ = self.predict(X_test[:10])
            latency = (time.time() - start_time) / 10 * 1000
            
            self.evaluation_metrics = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'latency_ms': latency,
                'test_samples': len(X_test)
            }
            
            logger.info(f"Model evaluation completed:")
            logger.info(f"  MSE: {mse:.4f}")
            logger.info(f"  MAE: {mae:.4f}")
            logger.info(f"  R²: {r2:.4f}")
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
    
    def schedule_maintenance(self, equipment_health_data: Dict[str, Any], 
                           maintenance_resources: Dict[str, Any],
                           production_schedule: Dict[str, Any]) -> Dict[str, Any]:
        """
        Schedule maintenance for equipment based on health data and constraints.
        
        Args:
            equipment_health_data: Health data for all equipment
            maintenance_resources: Available maintenance resources
            production_schedule: Production schedule constraints
            
        Returns:
            Dictionary with maintenance schedule
        """
        try:
            # Analyze equipment health and prioritize
            equipment_priorities = self._analyze_equipment_priorities(equipment_health_data)
            
            # Optimize maintenance schedule
            optimized_schedule = self._optimize_maintenance_schedule(
                equipment_priorities, maintenance_resources, production_schedule
            )
            
            # Calculate schedule metrics
            schedule_metrics = self._calculate_schedule_metrics(optimized_schedule)
            
            return {
                'maintenance_schedule': optimized_schedule,
                'schedule_metrics': schedule_metrics,
                'resource_utilization': self._calculate_resource_utilization(optimized_schedule, maintenance_resources),
                'risk_assessment': self._assess_schedule_risk(optimized_schedule, equipment_health_data)
            }
            
        except Exception as e:
            logger.error(f"Failed to schedule maintenance: {e}")
            raise
    
    def _analyze_equipment_priorities(self, equipment_health_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze equipment health and determine maintenance priorities.
        
        Args:
            equipment_health_data: Health data for all equipment
            
        Returns:
            List of equipment with priorities
        """
        priorities = []
        
        for equipment_id, health_data in equipment_health_data.items():
            # Create feature vector
            features = self._create_equipment_feature_vector(health_data)
            
            # Predict maintenance priority
            priority_score = self.predict(features.reshape(1, -1))[0]
            
            # Determine maintenance type
            maintenance_type = self._determine_maintenance_type(health_data, priority_score)
            
            # Estimate maintenance duration
            duration = self._estimate_maintenance_duration(equipment_id, maintenance_type)
            
            # Calculate urgency
            urgency = self._calculate_urgency(health_data, priority_score)
            
            priorities.append({
                'equipment_id': equipment_id,
                'priority_score': float(priority_score),
                'maintenance_type': maintenance_type,
                'estimated_duration_hours': duration,
                'urgency': urgency,
                'health_score': health_data.get('health_score', 0.5),
                'failure_probability': health_data.get('failure_probability', 0.0)
            })
        
        # Sort by priority score (descending)
        priorities.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return priorities
    
    def _create_equipment_feature_vector(self, health_data: Dict[str, Any]) -> np.ndarray:
        """
        Create feature vector from equipment health data.
        
        Args:
            health_data: Equipment health data
            
        Returns:
            Feature vector
        """
        feature_names = self.get_feature_names()
        features = np.zeros(len(feature_names))
        
        # Map health data to features
        for key, value in health_data.items():
            if key in feature_names:
                features[feature_names.index(key)] = value
        
        return features
    
    def _determine_maintenance_type(self, health_data: Dict[str, Any], priority_score: float) -> str:
        """
        Determine appropriate maintenance type based on health data and priority.
        
        Args:
            health_data: Equipment health data
            priority_score: Maintenance priority score
            
        Returns:
            Maintenance type
        """
        failure_probability = health_data.get('failure_probability', 0.0)
        health_score = health_data.get('health_score', 0.5)
        
        if failure_probability > 0.8 or priority_score > 0.9:
            return 'emergency'
        elif failure_probability > 0.6 or priority_score > 0.7:
            return 'corrective'
        elif health_score < 0.6 or priority_score > 0.5:
            return 'predictive'
        else:
            return 'preventive'
    
    def _estimate_maintenance_duration(self, equipment_id: str, maintenance_type: str) -> float:
        """
        Estimate maintenance duration based on equipment and maintenance type.
        
        Args:
            equipment_id: Equipment identifier
            maintenance_type: Type of maintenance
            
        Returns:
            Estimated duration in hours
        """
        # Base durations by maintenance type
        base_durations = {
            'preventive': 4.0,
            'predictive': 2.0,
            'corrective': 8.0,
            'emergency': 12.0
        }
        
        # Adjust based on equipment complexity
        equipment_complexity = {
            'laser': 1.5,
            'recoater': 1.0,
            'chamber': 2.0,
            'gas_system': 1.2
        }
        
        base_duration = base_durations.get(maintenance_type, 4.0)
        complexity_factor = equipment_complexity.get(equipment_id.split('_')[0], 1.0)
        
        return base_duration * complexity_factor
    
    def _calculate_urgency(self, health_data: Dict[str, Any], priority_score: float) -> str:
        """
        Calculate maintenance urgency.
        
        Args:
            health_data: Equipment health data
            priority_score: Maintenance priority score
            
        Returns:
            Urgency level
        """
        failure_probability = health_data.get('failure_probability', 0.0)
        
        if failure_probability > 0.8 or priority_score > 0.9:
            return 'critical'
        elif failure_probability > 0.6 or priority_score > 0.7:
            return 'high'
        elif failure_probability > 0.4 or priority_score > 0.5:
            return 'medium'
        else:
            return 'low'
    
    def _optimize_maintenance_schedule(self, equipment_priorities: List[Dict[str, Any]], 
                                     maintenance_resources: Dict[str, Any],
                                     production_schedule: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Optimize maintenance schedule considering resources and production constraints.
        
        Args:
            equipment_priorities: Equipment with priorities
            maintenance_resources: Available resources
            production_schedule: Production schedule
            
        Returns:
            Optimized maintenance schedule
        """
        schedule = []
        current_time = datetime.now()
        resource_availability = maintenance_resources.copy()
        
        for equipment in equipment_priorities:
            # Find optimal time slot
            optimal_time = self._find_optimal_time_slot(
                equipment, resource_availability, production_schedule, current_time
            )
            
            # Allocate resources
            allocated_resources = self._allocate_resources(
                equipment, resource_availability, optimal_time
            )
            
            if allocated_resources:
                schedule.append({
                    'equipment_id': equipment['equipment_id'],
                    'maintenance_type': equipment['maintenance_type'],
                    'scheduled_time': optimal_time,
                    'estimated_duration_hours': equipment['estimated_duration_hours'],
                    'allocated_resources': allocated_resources,
                    'priority_score': equipment['priority_score'],
                    'urgency': equipment['urgency']
                })
                
                # Update resource availability
                self._update_resource_availability(
                    resource_availability, allocated_resources, optimal_time, 
                    equipment['estimated_duration_hours']
                )
        
        return schedule
    
    def _find_optimal_time_slot(self, equipment: Dict[str, Any], 
                               resource_availability: Dict[str, Any],
                               production_schedule: Dict[str, Any],
                               current_time: datetime) -> datetime:
        """
        Find optimal time slot for maintenance.
        
        Args:
            equipment: Equipment information
            resource_availability: Available resources
            production_schedule: Production schedule
            
        Returns:
            Optimal maintenance time
        """
        # Start from current time
        candidate_time = current_time
        
        # For emergency maintenance, schedule immediately
        if equipment['maintenance_type'] == 'emergency':
            return candidate_time
        
        # For other types, find next available slot
        max_search_days = 30
        search_days = 0
        
        while search_days < max_search_days:
            # Check if resources are available
            if self._check_resource_availability(resource_availability, candidate_time, equipment['estimated_duration_hours']):
                # Check if production allows maintenance
                if self._check_production_compatibility(production_schedule, candidate_time, equipment['estimated_duration_hours']):
                    return candidate_time
            
            # Move to next day
            candidate_time += timedelta(days=1)
            search_days += 1
        
        # If no optimal slot found, return current time + 1 day
        return current_time + timedelta(days=1)
    
    def _check_resource_availability(self, resource_availability: Dict[str, Any], 
                                   maintenance_time: datetime, duration_hours: float) -> bool:
        """
        Check if resources are available for maintenance.
        
        Args:
            resource_availability: Available resources
            maintenance_time: Proposed maintenance time
            duration_hours: Maintenance duration
            
        Returns:
            True if resources are available
        """
        # Simplified check - in practice, this would be more sophisticated
        required_resources = ['technician', 'tools', 'parts']
        
        for resource in required_resources:
            if resource not in resource_availability:
                return False
            
            available_count = resource_availability[resource].get('count', 0)
            if available_count <= 0:
                return False
        
        return True
    
    def _check_production_compatibility(self, production_schedule: Dict[str, Any], 
                                      maintenance_time: datetime, duration_hours: float) -> bool:
        """
        Check if maintenance is compatible with production schedule.
        
        Args:
            production_schedule: Production schedule
            maintenance_time: Proposed maintenance time
            duration_hours: Maintenance duration
            
        Returns:
            True if compatible with production
        """
        # Simplified check - in practice, this would check actual production windows
        maintenance_end = maintenance_time + timedelta(hours=duration_hours)
        
        # Check if maintenance overlaps with critical production periods
        for production_period in production_schedule.get('critical_periods', []):
            start_time = production_period['start']
            end_time = production_period['end']
            
            if (maintenance_time < end_time and maintenance_end > start_time):
                return False
        
        return True
    
    def _allocate_resources(self, equipment: Dict[str, Any], 
                          resource_availability: Dict[str, Any],
                          maintenance_time: datetime) -> Optional[Dict[str, Any]]:
        """
        Allocate resources for maintenance.
        
        Args:
            equipment: Equipment information
            resource_availability: Available resources
            maintenance_time: Maintenance time
            
        Returns:
            Allocated resources or None if allocation failed
        """
        allocated = {}
        
        # Allocate technician
        if 'technician' in resource_availability and resource_availability['technician']['count'] > 0:
            allocated['technician'] = {
                'id': f"tech_{len(allocated)}",
                'skill_level': 'expert' if equipment['maintenance_type'] == 'emergency' else 'standard'
            }
            resource_availability['technician']['count'] -= 1
        else:
            return None
        
        # Allocate tools
        if 'tools' in resource_availability and resource_availability['tools']['count'] > 0:
            allocated['tools'] = {
                'set_id': f"toolset_{len(allocated)}",
                'specialized': equipment['maintenance_type'] in ['corrective', 'emergency']
            }
            resource_availability['tools']['count'] -= 1
        else:
            return None
        
        # Allocate parts (if needed)
        if equipment['maintenance_type'] in ['corrective', 'emergency']:
            if 'parts' in resource_availability and resource_availability['parts']['count'] > 0:
                allocated['parts'] = {
                    'kit_id': f"parts_{len(allocated)}",
                    'equipment_type': equipment['equipment_id'].split('_')[0]
                }
                resource_availability['parts']['count'] -= 1
        
        return allocated
    
    def _update_resource_availability(self, resource_availability: Dict[str, Any], 
                                    allocated_resources: Dict[str, Any],
                                    maintenance_time: datetime, duration_hours: float):
        """
        Update resource availability after allocation.
        
        Args:
            resource_availability: Resource availability
            allocated_resources: Allocated resources
            maintenance_time: Maintenance time
            duration_hours: Maintenance duration
        """
        # In practice, this would update a more sophisticated resource tracking system
        # For now, we just decrement counts
        pass
    
    def _calculate_schedule_metrics(self, schedule: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate schedule performance metrics.
        
        Args:
            schedule: Maintenance schedule
            
        Returns:
            Schedule metrics
        """
        if not schedule:
            return {'total_maintenance_hours': 0, 'equipment_count': 0}
        
        total_hours = sum(item['estimated_duration_hours'] for item in schedule)
        equipment_count = len(schedule)
        
        # Calculate priority distribution
        priority_distribution = {}
        for item in schedule:
            priority = item['maintenance_type']
            priority_distribution[priority] = priority_distribution.get(priority, 0) + 1
        
        return {
            'total_maintenance_hours': total_hours,
            'equipment_count': equipment_count,
            'average_duration_hours': total_hours / equipment_count,
            'priority_distribution': priority_distribution,
            'schedule_efficiency': self._calculate_schedule_efficiency(schedule)
        }
    
    def _calculate_schedule_efficiency(self, schedule: List[Dict[str, Any]]) -> float:
        """
        Calculate schedule efficiency score.
        
        Args:
            schedule: Maintenance schedule
            
        Returns:
            Efficiency score (0-1)
        """
        if not schedule:
            return 0.0
        
        # Factors affecting efficiency
        emergency_count = sum(1 for item in schedule if item['maintenance_type'] == 'emergency')
        preventive_count = sum(1 for item in schedule if item['maintenance_type'] == 'preventive')
        
        # Higher efficiency with more preventive maintenance
        efficiency = preventive_count / len(schedule)
        
        # Penalize emergency maintenance
        efficiency -= emergency_count * 0.2
        
        return max(0.0, min(1.0, efficiency))
    
    def _calculate_resource_utilization(self, schedule: List[Dict[str, Any]], 
                                      maintenance_resources: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate resource utilization.
        
        Args:
            schedule: Maintenance schedule
            maintenance_resources: Available resources
            
        Returns:
            Resource utilization metrics
        """
        utilization = {}
        
        for resource_type, resource_info in maintenance_resources.items():
            total_available = resource_info.get('count', 0)
            used_count = len(schedule)  # Simplified - each maintenance uses one of each resource
            
            utilization[resource_type] = {
                'total_available': total_available,
                'used': used_count,
                'utilization_rate': used_count / total_available if total_available > 0 else 0.0
            }
        
        return utilization
    
    def _assess_schedule_risk(self, schedule: List[Dict[str, Any]], 
                            equipment_health_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess risk associated with the maintenance schedule.
        
        Args:
            schedule: Maintenance schedule
            equipment_health_data: Equipment health data
            
        Returns:
            Risk assessment
        """
        # Calculate risk based on unscheduled equipment
        scheduled_equipment = {item['equipment_id'] for item in schedule}
        all_equipment = set(equipment_health_data.keys())
        unscheduled_equipment = all_equipment - scheduled_equipment
        
        # Calculate risk for unscheduled equipment
        unscheduled_risk = 0.0
        for equipment_id in unscheduled_equipment:
            health_data = equipment_health_data[equipment_id]
            failure_probability = health_data.get('failure_probability', 0.0)
            unscheduled_risk += failure_probability
        
        # Calculate risk for scheduled equipment (delayed maintenance)
        scheduled_risk = 0.0
        for item in schedule:
            equipment_id = item['equipment_id']
            health_data = equipment_health_data[equipment_id]
            failure_probability = health_data.get('failure_probability', 0.0)
            
            # Risk increases with delay
            delay_days = (item['scheduled_time'] - datetime.now()).days
            scheduled_risk += failure_probability * (1 + delay_days * 0.1)
        
        total_risk = unscheduled_risk + scheduled_risk
        
        return {
            'total_risk': total_risk,
            'unscheduled_risk': unscheduled_risk,
            'scheduled_risk': scheduled_risk,
            'unscheduled_equipment_count': len(unscheduled_equipment),
            'risk_level': 'high' if total_risk > 2.0 else 'medium' if total_risk > 1.0 else 'low'
        }
