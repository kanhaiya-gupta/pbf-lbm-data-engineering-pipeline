"""
Multi-Objective Optimizer Model

This module implements a multi-objective optimization model for balancing
speed, quality, material usage, and energy consumption in PBF-LB/M processes.
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


class MultiObjectiveOptimizer(BaseModel):
    """
    Multi-objective optimization model for PBF-LB/M processes.
    
    This model optimizes multiple objectives simultaneously:
    - Build speed (minimize build time)
    - Quality (maximize quality score)
    - Material usage (minimize material waste)
    - Energy consumption (minimize energy usage)
    - Cost (minimize total cost)
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the multi-objective optimizer.
        
        Args:
            config_manager: Configuration manager instance
        """
        super().__init__('multi_objective_optimizer', config_manager)
        self.model_type = self.model_info.get('algorithm', 'neural_network')
        self.objective_weights = self.config.get('objectives', {}).get('weights', {
            'speed': 0.25,
            'quality': 0.25,
            'material_usage': 0.25,
            'energy_consumption': 0.25
        })
        
        logger.info(f"Initialized MultiObjectiveOptimizer with algorithm: {self.model_type}")
    
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
        """Build neural network model for multi-objective optimization."""
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
        
        # Output layer (multiple objectives)
        num_objectives = len(self.objective_weights)
        model.add(Dense(num_objectives, activation='linear'))
        
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
            X_train: Training features
            y_train: Training targets (multiple objectives)
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
                patience=self.training_config.get('early_stopping', {}).get('patience', 20),
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.training_config.get('batch_size', 32),
            epochs=self.training_config.get('epochs', 200),
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
            Predictions array with multiple objectives
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            predictions = self.model.predict(X, verbose=0)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            raise
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Evaluation metrics dictionary
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before evaluation")
            
            # Make predictions
            predictions = self.predict(X_test)
            
            # Calculate metrics for each objective
            metrics = {}
            objective_names = list(self.objective_weights.keys())
            
            for i, obj_name in enumerate(objective_names):
                mse = mean_squared_error(y_test[:, i], predictions[:, i])
                mae = mean_absolute_error(y_test[:, i], predictions[:, i])
                r2 = r2_score(y_test[:, i], predictions[:, i])
                
                metrics.update({
                    f'{obj_name}_mse': mse,
                    f'{obj_name}_mae': mae,
                    f'{obj_name}_r2': r2
                })
            
            # Overall metrics
            overall_mse = mean_squared_error(y_test, predictions)
            overall_mae = mean_absolute_error(y_test, predictions)
            overall_r2 = r2_score(y_test, predictions)
            
            metrics.update({
                'overall_mse': overall_mse,
                'overall_mae': overall_mae,
                'overall_r2': overall_r2
            })
            
            # Calculate latency
            start_time = time.time()
            _ = self.predict(X_test[:10])
            latency = (time.time() - start_time) / 10 * 1000
            
            metrics['latency_ms'] = latency
            
            self.evaluation_metrics = metrics
            
            logger.info(f"Model evaluation completed:")
            logger.info(f"  Overall MSE: {overall_mse:.4f}")
            logger.info(f"  Overall MAE: {overall_mae:.4f}")
            logger.info(f"  Overall R²: {overall_r2:.4f}")
            logger.info(f"  Latency: {latency:.2f} ms")
            
            return metrics
            
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
    
    def optimize_parameters(self, constraints: Dict[str, Tuple[float, float]], 
                           objectives: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Optimize process parameters for multiple objectives.
        
        Args:
            constraints: Parameter constraints (min, max) for each parameter
            objectives: Objective weights (optional, uses default if None)
            
        Returns:
            Dictionary with optimized parameters
        """
        try:
            if objectives is None:
                objectives = self.objective_weights
            
            # Define objective function
            def objective_function(params):
                # Convert parameters to feature vector
                features = self._params_to_features(params, constraints)
                
                # Predict objectives
                objectives_pred = self.predict(features.reshape(1, -1))[0]
                
                # Calculate weighted sum (minimize)
                weighted_sum = 0
                for i, (obj_name, weight) in enumerate(objectives.items()):
                    weighted_sum += weight * objectives_pred[i]
                
                return weighted_sum
            
            # Define constraints
            bounds = list(constraints.values())
            
            # Initial guess (middle of bounds)
            x0 = [(bounds[i][0] + bounds[i][1]) / 2 for i in range(len(bounds))]
            
            # Optimize
            result = minimize(
                objective_function,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )
            
            # Map result to parameter names
            param_names = list(constraints.keys())
            optimized_params = {name: float(result.x[i]) for i, name in enumerate(param_names)}
            
            return optimized_params
            
        except Exception as e:
            logger.error(f"Failed to optimize parameters: {e}")
            raise
    
    def _params_to_features(self, params: List[float], constraints: Dict[str, Tuple[float, float]]) -> np.ndarray:
        """
        Convert parameter values to feature vector.
        
        Args:
            params: Parameter values
            constraints: Parameter constraints
            
        Returns:
            Feature vector
        """
        feature_names = self.get_feature_names()
        features = np.zeros(len(feature_names))
        
        param_names = list(constraints.keys())
        
        for i, param_name in enumerate(param_names):
            if param_name in feature_names:
                features[feature_names.index(param_name)] = params[i]
        
        return features
    
    def get_pareto_front(self, constraints: Dict[str, Tuple[float, float]], 
                        n_points: int = 100) -> List[Dict[str, float]]:
        """
        Get Pareto front for multi-objective optimization.
        
        Args:
            constraints: Parameter constraints
            n_points: Number of points on Pareto front
            
        Returns:
            List of Pareto optimal solutions
        """
        try:
            pareto_solutions = []
            
            # Generate random parameter combinations
            for _ in range(n_points):
                # Random parameters within constraints
                params = {}
                for param_name, (min_val, max_val) in constraints.items():
                    params[param_name] = np.random.uniform(min_val, max_val)
                
                # Convert to feature vector
                features = self._params_to_features(list(params.values()), constraints)
                
                # Predict objectives
                objectives = self.predict(features.reshape(1, -1))[0]
                
                # Store solution
                solution = {
                    'parameters': params,
                    'objectives': {
                        obj_name: float(objectives[i]) 
                        for i, obj_name in enumerate(self.objective_weights.keys())
                    }
                }
                pareto_solutions.append(solution)
            
            # Filter Pareto optimal solutions
            pareto_optimal = self._filter_pareto_optimal(pareto_solutions)
            
            return pareto_optimal
            
        except Exception as e:
            logger.error(f"Failed to get Pareto front: {e}")
            raise
    
    def _filter_pareto_optimal(self, solutions: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """
        Filter Pareto optimal solutions.
        
        Args:
            solutions: List of solutions with parameters and objectives
            
        Returns:
            List of Pareto optimal solutions
        """
        pareto_optimal = []
        
        for solution in solutions:
            is_pareto = True
            objectives = solution['objectives']
            
            for other_solution in solutions:
                if solution == other_solution:
                    continue
                
                other_objectives = other_solution['objectives']
                
                # Check if other solution dominates this one
                dominates = True
                for obj_name in objectives.keys():
                    if other_objectives[obj_name] > objectives[obj_name]:
                        dominates = False
                        break
                
                if dominates:
                    is_pareto = False
                    break
            
            if is_pareto:
                pareto_optimal.append(solution)
        
        return pareto_optimal
