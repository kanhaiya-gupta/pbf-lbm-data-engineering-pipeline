"""
Root Cause Analyzer Model

This module implements a model for analyzing root causes of defects
in PBF-LB/M manufacturing processes.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

from ..base_model import BaseModel

logger = logging.getLogger(__name__)


class RootCauseAnalyzer(BaseModel):
    """
    Model for analyzing root causes of defects in PBF-LB/M processes.
    
    This model analyzes:
    - Process parameter root causes
    - Material-related root causes
    - Equipment-related root causes
    - Environmental root causes
    - Design-related root causes
    - Operator-related root causes
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the root cause analyzer.
        
        Args:
            config_manager: Configuration manager instance
        """
        super().__init__('root_cause_analyzer', config_manager)
        self.model_type = self.model_info.get('algorithm', 'neural_network')
        self.root_cause_categories = [
            'process_parameters', 'material_issues', 'equipment_problems',
            'environmental_factors', 'design_issues', 'operator_errors'
        ]
        
        # Initialize root cause knowledge base
        self.root_cause_knowledge = self._initialize_root_cause_knowledge()
        
        logger.info(f"Initialized RootCauseAnalyzer with algorithm: {self.model_type}")
    
    def _initialize_root_cause_knowledge(self) -> Dict[str, Any]:
        """
        Initialize root cause knowledge base.
        
        Returns:
            Dictionary with root cause knowledge
        """
        return {
            'process_parameters': {
                'laser_power': {
                    'too_high': ['pore_keyhole', 'crack_hot', 'overheating'],
                    'too_low': ['lack_of_fusion', 'pore_spherical', 'incomplete_melting']
                },
                'scan_speed': {
                    'too_high': ['lack_of_fusion', 'pore_irregular', 'incomplete_melting'],
                    'too_low': ['overheating', 'crack_hot', 'warping']
                },
                'hatch_spacing': {
                    'too_wide': ['lack_of_fusion', 'pore_irregular'],
                    'too_narrow': ['overheating', 'crack_hot']
                },
                'layer_thickness': {
                    'too_thick': ['lack_of_fusion', 'pore_irregular'],
                    'too_thin': ['overheating', 'crack_hot']
                }
            },
            'material_issues': {
                'powder_quality': {
                    'poor_quality': ['pore_spherical', 'contamination', 'lack_of_fusion'],
                    'contamination': ['contamination', 'pore_irregular'],
                    'moisture': ['pore_spherical', 'contamination']
                },
                'powder_distribution': {
                    'uneven': ['lack_of_fusion', 'pore_irregular'],
                    'insufficient': ['lack_of_fusion', 'pore_spherical']
                },
                'powder_size': {
                    'too_large': ['lack_of_fusion', 'pore_irregular'],
                    'too_small': ['overheating', 'crack_hot']
                }
            },
            'equipment_problems': {
                'laser_system': {
                    'beam_quality': ['lack_of_fusion', 'pore_irregular'],
                    'power_stability': ['pore_spherical', 'crack_hot'],
                    'focus_position': ['lack_of_fusion', 'overheating']
                },
                'recoater_system': {
                    'movement_accuracy': ['lack_of_fusion', 'pore_irregular'],
                    'wear': ['surface_roughness', 'pore_irregular']
                },
                'chamber_system': {
                    'atmosphere': ['contamination', 'pore_spherical'],
                    'temperature': ['crack_hot', 'warping'],
                    'pressure': ['pore_spherical', 'contamination']
                }
            },
            'environmental_factors': {
                'temperature': {
                    'too_high': ['overheating', 'crack_hot'],
                    'too_low': ['crack_cold', 'lack_of_fusion']
                },
                'humidity': {
                    'too_high': ['contamination', 'pore_spherical'],
                    'too_low': ['static_electricity', 'powder_distribution']
                },
                'vibration': {
                    'excessive': ['lack_of_fusion', 'pore_irregular']
                }
            },
            'design_issues': {
                'overhang_angles': {
                    'too_steep': ['overhang', 'lack_of_fusion']
                },
                'wall_thickness': {
                    'too_thin': ['crack_stress', 'warping']
                },
                'support_design': {
                    'insufficient': ['warping', 'crack_stress']
                }
            },
            'operator_errors': {
                'parameter_setting': {
                    'incorrect': ['lack_of_fusion', 'overheating', 'pore_irregular']
                },
                'powder_handling': {
                    'contamination': ['contamination', 'pore_spherical']
                },
                'build_preparation': {
                    'inadequate': ['lack_of_fusion', 'pore_irregular']
                }
            }
        }
    
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
        model.add(Dense(512, activation='relu', input_dim=input_dim))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        # Hidden layers
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.1))
        
        # Output layer (root cause probabilities)
        model.add(Dense(len(self.root_cause_categories), activation='softmax'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.training_config.get('learning_rate', 0.001)),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def _build_random_forest(self) -> RandomForestClassifier:
        """Build random forest model."""
        return RandomForestClassifier(
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
            X_train: Training features (defect characteristics, process data)
            y_train: Training targets (root cause categories)
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
            epochs=self.training_config.get('epochs', 100),
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
        train_accuracy = accuracy_score(y_train, train_pred)
        train_precision = precision_score(y_train, train_pred, average='weighted', zero_division=0)
        train_recall = recall_score(y_train, train_pred, average='weighted', zero_division=0)
        train_f1 = f1_score(y_train, train_pred, average='weighted', zero_division=0)
        
        history = {
            'accuracy': [train_accuracy],
            'precision': [train_precision],
            'recall': [train_recall],
            'f1_score': [train_f1]
        }
        
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
            val_precision = precision_score(y_val, val_pred, average='weighted', zero_division=0)
            val_recall = recall_score(y_val, val_pred, average='weighted', zero_division=0)
            val_f1 = f1_score(y_val, val_pred, average='weighted', zero_division=0)
            
            history.update({
                'val_accuracy': [val_accuracy],
                'val_precision': [val_precision],
                'val_recall': [val_recall],
                'val_f1_score': [val_f1]
            })
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features (defect characteristics, process data)
            
        Returns:
            Predictions array (root cause probabilities)
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            predictions = self.model.predict(X, verbose=0)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            raise
    
    def predict_root_cause(self, X: np.ndarray) -> np.ndarray:
        """
        Predict root cause category (hard classification).
        
        Args:
            X: Input features
            
        Returns:
            Root cause category predictions
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            predictions = self.predict(X)
            class_predictions = np.argmax(predictions, axis=1)
            
            return class_predictions
            
        except Exception as e:
            logger.error(f"Failed to predict root cause: {e}")
            raise
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets (root cause categories)
            
        Returns:
            Evaluation metrics dictionary
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before evaluation")
            
            # Make predictions
            predictions = self.predict(X_test)
            class_predictions = self.predict_root_cause(X_test)
            
            # Convert y_test to class labels if needed
            if len(y_test.shape) > 1:
                y_test_classes = np.argmax(y_test, axis=1)
            else:
                y_test_classes = y_test
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_classes, class_predictions)
            precision = precision_score(y_test_classes, class_predictions, average='weighted', zero_division=0)
            recall = recall_score(y_test_classes, class_predictions, average='weighted', zero_division=0)
            f1 = f1_score(y_test_classes, class_predictions, average='weighted', zero_division=0)
            
            # Calculate per-class metrics
            per_class_metrics = self._calculate_per_class_metrics(y_test_classes, class_predictions)
            
            # Calculate latency
            start_time = time.time()
            _ = self.predict(X_test[:10])
            latency = (time.time() - start_time) / 10 * 1000
            
            self.evaluation_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'per_class_metrics': per_class_metrics,
                'latency_ms': latency,
                'test_samples': len(X_test)
            }
            
            logger.info(f"Model evaluation completed:")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  F1-Score: {f1:.4f}")
            logger.info(f"  Latency: {latency:.2f} ms")
            
            return self.evaluation_metrics
            
        except Exception as e:
            logger.error(f"Failed to evaluate model: {e}")
            raise
    
    def _calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Calculate per-class metrics.
        
        Args:
            y_true: True class labels
            y_pred: Predicted class labels
            
        Returns:
            Dictionary with per-class metrics
        """
        per_class_metrics = {}
        
        for i, class_name in enumerate(self.root_cause_categories):
            # Create binary labels for this class
            y_true_binary = (y_true == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)
            
            # Calculate metrics
            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            
            per_class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        return per_class_metrics
    
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
    
    def analyze_root_cause(self, defect_characteristics: Dict[str, Any], 
                          process_data: Dict[str, Any],
                          equipment_data: Dict[str, Any],
                          environmental_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze root cause of defects.
        
        Args:
            defect_characteristics: Defect characteristics
            process_data: Process data
            equipment_data: Equipment data
            environmental_data: Environmental data
            
        Returns:
            Dictionary with root cause analysis results
        """
        try:
            # Create feature vector from inputs
            features = self._create_feature_vector(defect_characteristics, process_data, equipment_data, environmental_data)
            
            # Make prediction
            predictions = self.predict(features.reshape(1, -1))
            root_cause_probabilities = predictions[0]
            predicted_category = np.argmax(root_cause_probabilities)
            confidence = root_cause_probabilities[predicted_category]
            
            # Analyze root cause details
            root_cause_analysis = self._analyze_root_cause_details(
                predicted_category, defect_characteristics, process_data, equipment_data, environmental_data
            )
            
            # Generate specific root cause hypotheses
            root_cause_hypotheses = self._generate_root_cause_hypotheses(
                predicted_category, defect_characteristics, process_data, equipment_data, environmental_data
            )
            
            # Assess root cause confidence
            confidence_assessment = self._assess_root_cause_confidence(
                root_cause_probabilities, defect_characteristics, process_data
            )
            
            # Generate corrective actions
            corrective_actions = self._generate_corrective_actions(
                predicted_category, root_cause_analysis, defect_characteristics
            )
            
            # Calculate root cause impact
            impact_assessment = self._calculate_root_cause_impact(
                predicted_category, defect_characteristics, process_data
            )
            
            return {
                'root_cause_analysis': {
                    'predicted_category': self.root_cause_categories[predicted_category],
                    'confidence': float(confidence),
                    'category_probabilities': {self.root_cause_categories[i]: float(prob) for i, prob in enumerate(root_cause_probabilities)}
                },
                'root_cause_details': root_cause_analysis,
                'root_cause_hypotheses': root_cause_hypotheses,
                'confidence_assessment': confidence_assessment,
                'corrective_actions': corrective_actions,
                'impact_assessment': impact_assessment,
                'analysis_summary': self._generate_analysis_summary(predicted_category, confidence, root_cause_analysis)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze root cause: {e}")
            raise
    
    def _create_feature_vector(self, defect_characteristics: Dict[str, Any], 
                              process_data: Dict[str, Any],
                              equipment_data: Dict[str, Any],
                              environmental_data: Dict[str, Any]) -> np.ndarray:
        """
        Create feature vector from input parameters.
        
        Args:
            defect_characteristics: Defect characteristics
            process_data: Process data
            equipment_data: Equipment data
            environmental_data: Environmental data
            
        Returns:
            Feature vector
        """
        feature_names = self.get_feature_names()
        features = np.zeros(len(feature_names))
        
        # Combine all input dictionaries
        all_inputs = {**defect_characteristics, **process_data, **equipment_data, **environmental_data}
        
        # Map inputs to features
        for key, value in all_inputs.items():
            if key in feature_names:
                features[feature_names.index(key)] = value
        
        return features
    
    def _analyze_root_cause_details(self, predicted_category: int, 
                                  defect_characteristics: Dict[str, Any],
                                  process_data: Dict[str, Any],
                                  equipment_data: Dict[str, Any],
                                  environmental_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze detailed root cause information.
        
        Args:
            predicted_category: Predicted root cause category
            defect_characteristics: Defect characteristics
            process_data: Process data
            equipment_data: Equipment data
            environmental_data: Environmental data
            
        Returns:
            Dictionary with detailed root cause analysis
        """
        category_name = self.root_cause_categories[predicted_category]
        defect_type = defect_characteristics.get('defect_type', 'unknown')
        
        # Get specific root cause factors
        specific_factors = self._identify_specific_factors(category_name, defect_type, process_data, equipment_data, environmental_data)
        
        # Analyze contributing factors
        contributing_factors = self._analyze_contributing_factors(category_name, process_data, equipment_data, environmental_data)
        
        # Assess factor severity
        factor_severity = self._assess_factor_severity(specific_factors, contributing_factors)
        
        # Calculate root cause probability
        root_cause_probability = self._calculate_root_cause_probability(specific_factors, contributing_factors)
        
        return {
            'category': category_name,
            'specific_factors': specific_factors,
            'contributing_factors': contributing_factors,
            'factor_severity': factor_severity,
            'root_cause_probability': root_cause_probability,
            'analysis_depth': self._assess_analysis_depth(specific_factors, contributing_factors)
        }
    
    def _identify_specific_factors(self, category_name: str, defect_type: str, 
                                 process_data: Dict[str, Any],
                                 equipment_data: Dict[str, Any],
                                 environmental_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify specific factors contributing to the root cause.
        
        Args:
            category_name: Root cause category name
            defect_type: Defect type
            process_data: Process data
            equipment_data: Equipment data
            environmental_data: Environmental data
            
        Returns:
            List of specific factors
        """
        specific_factors = []
        
        if category_name == 'process_parameters':
            # Analyze process parameters
            for param, value in process_data.items():
                if param in self.root_cause_knowledge['process_parameters']:
                    param_analysis = self.root_cause_knowledge['process_parameters'][param]
                    
                    # Check if parameter value is problematic
                    if 'too_high' in param_analysis and defect_type in param_analysis['too_high']:
                        if self._is_parameter_too_high(param, value):
                            specific_factors.append({
                                'factor': param,
                                'issue': 'too_high',
                                'value': value,
                                'impact': 'high',
                                'description': f'{param} is too high, causing {defect_type}'
                            })
                    
                    if 'too_low' in param_analysis and defect_type in param_analysis['too_low']:
                        if self._is_parameter_too_low(param, value):
                            specific_factors.append({
                                'factor': param,
                                'issue': 'too_low',
                                'value': value,
                                'impact': 'high',
                                'description': f'{param} is too low, causing {defect_type}'
                            })
        
        elif category_name == 'material_issues':
            # Analyze material issues
            for material_param, value in process_data.items():
                if material_param in self.root_cause_knowledge['material_issues']:
                    material_analysis = self.root_cause_knowledge['material_issues'][material_param]
                    
                    for issue, defect_types in material_analysis.items():
                        if defect_type in defect_types:
                            specific_factors.append({
                                'factor': material_param,
                                'issue': issue,
                                'value': value,
                                'impact': 'medium',
                                'description': f'{material_param} has {issue}, causing {defect_type}'
                            })
        
        elif category_name == 'equipment_problems':
            # Analyze equipment problems
            for equipment_param, value in equipment_data.items():
                if equipment_param in self.root_cause_knowledge['equipment_problems']:
                    equipment_analysis = self.root_cause_knowledge['equipment_problems'][equipment_param]
                    
                    for issue, defect_types in equipment_analysis.items():
                        if defect_type in defect_types:
                            specific_factors.append({
                                'factor': equipment_param,
                                'issue': issue,
                                'value': value,
                                'impact': 'high',
                                'description': f'{equipment_param} has {issue}, causing {defect_type}'
                            })
        
        elif category_name == 'environmental_factors':
            # Analyze environmental factors
            for env_param, value in environmental_data.items():
                if env_param in self.root_cause_knowledge['environmental_factors']:
                    env_analysis = self.root_cause_knowledge['environmental_factors'][env_param]
                    
                    for issue, defect_types in env_analysis.items():
                        if defect_type in defect_types:
                            specific_factors.append({
                                'factor': env_param,
                                'issue': issue,
                                'value': value,
                                'impact': 'medium',
                                'description': f'{env_param} is {issue}, causing {defect_type}'
                            })
        
        return specific_factors
    
    def _is_parameter_too_high(self, param: str, value: float) -> bool:
        """Check if parameter value is too high."""
        # Define normal ranges for parameters
        normal_ranges = {
            'laser_power': (200, 600),
            'scan_speed': (500, 2000),
            'hatch_spacing': (0.05, 0.2),
            'layer_thickness': (0.02, 0.1)
        }
        
        if param in normal_ranges:
            min_val, max_val = normal_ranges[param]
            return value > max_val
        
        return False
    
    def _is_parameter_too_low(self, param: str, value: float) -> bool:
        """Check if parameter value is too low."""
        # Define normal ranges for parameters
        normal_ranges = {
            'laser_power': (200, 600),
            'scan_speed': (500, 2000),
            'hatch_spacing': (0.05, 0.2),
            'layer_thickness': (0.02, 0.1)
        }
        
        if param in normal_ranges:
            min_val, max_val = normal_ranges[param]
            return value < min_val
        
        return False
    
    def _analyze_contributing_factors(self, category_name: str, 
                                    process_data: Dict[str, Any],
                                    equipment_data: Dict[str, Any],
                                    environmental_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze contributing factors to the root cause.
        
        Args:
            category_name: Root cause category name
            process_data: Process data
            equipment_data: Equipment data
            environmental_data: Environmental data
            
        Returns:
            List of contributing factors
        """
        contributing_factors = []
        
        # Analyze process data for contributing factors
        for param, value in process_data.items():
            if self._is_contributing_factor(param, value, category_name):
                contributing_factors.append({
                    'factor': param,
                    'value': value,
                    'contribution': 'medium',
                    'description': f'{param} may be contributing to {category_name} issues'
                })
        
        # Analyze equipment data for contributing factors
        for param, value in equipment_data.items():
            if self._is_contributing_factor(param, value, category_name):
                contributing_factors.append({
                    'factor': param,
                    'value': value,
                    'contribution': 'high',
                    'description': f'{param} may be contributing to {category_name} issues'
                })
        
        # Analyze environmental data for contributing factors
        for param, value in environmental_data.items():
            if self._is_contributing_factor(param, value, category_name):
                contributing_factors.append({
                    'factor': param,
                    'value': value,
                    'contribution': 'low',
                    'description': f'{param} may be contributing to {category_name} issues'
                })
        
        return contributing_factors
    
    def _is_contributing_factor(self, param: str, value: float, category_name: str) -> bool:
        """Check if a parameter is a contributing factor."""
        # Define thresholds for contributing factors
        contributing_thresholds = {
            'laser_power': (150, 700),
            'scan_speed': (300, 2500),
            'hatch_spacing': (0.03, 0.25),
            'layer_thickness': (0.01, 0.15),
            'temperature': (15, 35),
            'humidity': (20, 80)
        }
        
        if param in contributing_thresholds:
            min_val, max_val = contributing_thresholds[param]
            return not (min_val <= value <= max_val)
        
        return False
    
    def _assess_factor_severity(self, specific_factors: List[Dict[str, Any]], 
                              contributing_factors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assess severity of identified factors.
        
        Args:
            specific_factors: Specific factors
            contributing_factors: Contributing factors
            
        Returns:
            Dictionary with factor severity assessment
        """
        high_severity_count = sum(1 for factor in specific_factors if factor['impact'] == 'high')
        medium_severity_count = sum(1 for factor in specific_factors if factor['impact'] == 'medium')
        low_severity_count = sum(1 for factor in specific_factors if factor['impact'] == 'low')
        
        total_factors = len(specific_factors) + len(contributing_factors)
        
        if total_factors == 0:
            severity_level = 'unknown'
        elif high_severity_count > 0:
            severity_level = 'high'
        elif medium_severity_count > 0:
            severity_level = 'medium'
        else:
            severity_level = 'low'
        
        return {
            'severity_level': severity_level,
            'high_severity_count': high_severity_count,
            'medium_severity_count': medium_severity_count,
            'low_severity_count': low_severity_count,
            'total_factors': total_factors,
            'severity_score': (high_severity_count * 3 + medium_severity_count * 2 + low_severity_count) / max(1, total_factors)
        }
    
    def _calculate_root_cause_probability(self, specific_factors: List[Dict[str, Any]], 
                                        contributing_factors: List[Dict[str, Any]]) -> float:
        """
        Calculate probability of root cause identification.
        
        Args:
            specific_factors: Specific factors
            contributing_factors: Contributing factors
            
        Returns:
            Root cause probability
        """
        if not specific_factors:
            return 0.3  # Low probability if no specific factors
        
        # Base probability on number and impact of specific factors
        base_probability = 0.5
        
        # Increase probability based on specific factors
        for factor in specific_factors:
            if factor['impact'] == 'high':
                base_probability += 0.2
            elif factor['impact'] == 'medium':
                base_probability += 0.1
        
        # Increase probability based on contributing factors
        base_probability += min(0.2, len(contributing_factors) * 0.05)
        
        return min(1.0, base_probability)
    
    def _assess_analysis_depth(self, specific_factors: List[Dict[str, Any]], 
                             contributing_factors: List[Dict[str, Any]]) -> str:
        """
        Assess depth of root cause analysis.
        
        Args:
            specific_factors: Specific factors
            contributing_factors: Contributing factors
            
        Returns:
            Analysis depth level
        """
        total_factors = len(specific_factors) + len(contributing_factors)
        
        if total_factors >= 5:
            return 'comprehensive'
        elif total_factors >= 3:
            return 'moderate'
        elif total_factors >= 1:
            return 'basic'
        else:
            return 'insufficient'
    
    def _generate_root_cause_hypotheses(self, predicted_category: int, 
                                      defect_characteristics: Dict[str, Any],
                                      process_data: Dict[str, Any],
                                      equipment_data: Dict[str, Any],
                                      environmental_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate specific root cause hypotheses.
        
        Args:
            predicted_category: Predicted root cause category
            defect_characteristics: Defect characteristics
            process_data: Process data
            equipment_data: Equipment data
            environmental_data: Environmental data
            
        Returns:
            List of root cause hypotheses
        """
        hypotheses = []
        category_name = self.root_cause_categories[predicted_category]
        defect_type = defect_characteristics.get('defect_type', 'unknown')
        
        if category_name == 'process_parameters':
            hypotheses.extend(self._generate_process_parameter_hypotheses(defect_type, process_data))
        elif category_name == 'material_issues':
            hypotheses.extend(self._generate_material_issue_hypotheses(defect_type, process_data))
        elif category_name == 'equipment_problems':
            hypotheses.extend(self._generate_equipment_problem_hypotheses(defect_type, equipment_data))
        elif category_name == 'environmental_factors':
            hypotheses.extend(self._generate_environmental_factor_hypotheses(defect_type, environmental_data))
        elif category_name == 'design_issues':
            hypotheses.extend(self._generate_design_issue_hypotheses(defect_type, process_data))
        elif category_name == 'operator_errors':
            hypotheses.extend(self._generate_operator_error_hypotheses(defect_type, process_data))
        
        return hypotheses
    
    def _generate_process_parameter_hypotheses(self, defect_type: str, process_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate process parameter hypotheses."""
        hypotheses = []
        
        for param, value in process_data.items():
            if param in self.root_cause_knowledge['process_parameters']:
                param_analysis = self.root_cause_knowledge['process_parameters'][param]
                
                if 'too_high' in param_analysis and defect_type in param_analysis['too_high']:
                    if self._is_parameter_too_high(param, value):
                        hypotheses.append({
                            'hypothesis': f'{param} is too high',
                            'evidence': f'{param} = {value}',
                            'confidence': 0.8,
                            'description': f'High {param} is causing {defect_type}'
                        })
                
                if 'too_low' in param_analysis and defect_type in param_analysis['too_low']:
                    if self._is_parameter_too_low(param, value):
                        hypotheses.append({
                            'hypothesis': f'{param} is too low',
                            'evidence': f'{param} = {value}',
                            'confidence': 0.8,
                            'description': f'Low {param} is causing {defect_type}'
                        })
        
        return hypotheses
    
    def _generate_material_issue_hypotheses(self, defect_type: str, process_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate material issue hypotheses."""
        hypotheses = []
        
        for material_param, value in process_data.items():
            if material_param in self.root_cause_knowledge['material_issues']:
                material_analysis = self.root_cause_knowledge['material_issues'][material_param]
                
                for issue, defect_types in material_analysis.items():
                    if defect_type in defect_types:
                        hypotheses.append({
                            'hypothesis': f'{material_param} has {issue}',
                            'evidence': f'{material_param} = {value}',
                            'confidence': 0.7,
                            'description': f'{material_param} {issue} is causing {defect_type}'
                        })
        
        return hypotheses
    
    def _generate_equipment_problem_hypotheses(self, defect_type: str, equipment_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate equipment problem hypotheses."""
        hypotheses = []
        
        for equipment_param, value in equipment_data.items():
            if equipment_param in self.root_cause_knowledge['equipment_problems']:
                equipment_analysis = self.root_cause_knowledge['equipment_problems'][equipment_param]
                
                for issue, defect_types in equipment_analysis.items():
                    if defect_type in defect_types:
                        hypotheses.append({
                            'hypothesis': f'{equipment_param} has {issue}',
                            'evidence': f'{equipment_param} = {value}',
                            'confidence': 0.9,
                            'description': f'{equipment_param} {issue} is causing {defect_type}'
                        })
        
        return hypotheses
    
    def _generate_environmental_factor_hypotheses(self, defect_type: str, environmental_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate environmental factor hypotheses."""
        hypotheses = []
        
        for env_param, value in environmental_data.items():
            if env_param in self.root_cause_knowledge['environmental_factors']:
                env_analysis = self.root_cause_knowledge['environmental_factors'][env_param]
                
                for issue, defect_types in env_analysis.items():
                    if defect_type in defect_types:
                        hypotheses.append({
                            'hypothesis': f'{env_param} is {issue}',
                            'evidence': f'{env_param} = {value}',
                            'confidence': 0.6,
                            'description': f'{env_param} {issue} is causing {defect_type}'
                        })
        
        return hypotheses
    
    def _generate_design_issue_hypotheses(self, defect_type: str, process_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate design issue hypotheses."""
        hypotheses = []
        
        # Check for design-related parameters
        if 'overhang_angle' in process_data:
            overhang_angle = process_data['overhang_angle']
            if overhang_angle > 45 and defect_type in ['overhang', 'lack_of_fusion']:
                hypotheses.append({
                    'hypothesis': 'Overhang angle is too steep',
                    'evidence': f'overhang_angle = {overhang_angle}°',
                    'confidence': 0.8,
                    'description': f'Steep overhang angle is causing {defect_type}'
                })
        
        if 'wall_thickness' in process_data:
            wall_thickness = process_data['wall_thickness']
            if wall_thickness < 0.5 and defect_type in ['crack_stress', 'warping']:
                hypotheses.append({
                    'hypothesis': 'Wall thickness is too thin',
                    'evidence': f'wall_thickness = {wall_thickness}mm',
                    'confidence': 0.7,
                    'description': f'Thin wall thickness is causing {defect_type}'
                })
        
        return hypotheses
    
    def _generate_operator_error_hypotheses(self, defect_type: str, process_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate operator error hypotheses."""
        hypotheses = []
        
        # Check for operator-related issues
        if 'parameter_setting' in process_data:
            parameter_setting = process_data['parameter_setting']
            if parameter_setting == 'incorrect':
                hypotheses.append({
                    'hypothesis': 'Incorrect parameter setting',
                    'evidence': 'parameter_setting = incorrect',
                    'confidence': 0.9,
                    'description': f'Incorrect parameter setting is causing {defect_type}'
                })
        
        if 'powder_handling' in process_data:
            powder_handling = process_data['powder_handling']
            if powder_handling == 'contamination':
                hypotheses.append({
                    'hypothesis': 'Powder handling contamination',
                    'evidence': 'powder_handling = contamination',
                    'confidence': 0.8,
                    'description': f'Powder handling contamination is causing {defect_type}'
                })
        
        return hypotheses
    
    def _assess_root_cause_confidence(self, root_cause_probabilities: np.ndarray, 
                                    defect_characteristics: Dict[str, Any],
                                    process_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess confidence in root cause analysis.
        
        Args:
            root_cause_probabilities: Root cause probabilities
            defect_characteristics: Defect characteristics
            process_data: Process data
            
        Returns:
            Dictionary with confidence assessment
        """
        max_probability = np.max(root_cause_probabilities)
        probability_entropy = -np.sum(root_cause_probabilities * np.log(root_cause_probabilities + 1e-10))
        
        # Assess data completeness
        required_fields = ['defect_type', 'defect_size', 'defect_location']
        completeness = sum(1 for field in required_fields if field in defect_characteristics) / len(required_fields)
        
        # Assess process data completeness
        process_completeness = len(process_data) / 10  # Assume 10 is complete
        
        # Calculate overall confidence
        overall_confidence = (max_probability + (1 - probability_entropy) + completeness + process_completeness) / 4
        
        return {
            'overall_confidence': float(overall_confidence),
            'max_probability': float(max_probability),
            'probability_entropy': float(probability_entropy),
            'data_completeness': float(completeness),
            'process_completeness': float(process_completeness),
            'confidence_level': 'high' if overall_confidence > 0.8 else 'medium' if overall_confidence > 0.6 else 'low'
        }
    
    def _generate_corrective_actions(self, predicted_category: int, 
                                   root_cause_analysis: Dict[str, Any],
                                   defect_characteristics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate corrective actions based on root cause analysis.
        
        Args:
            predicted_category: Predicted root cause category
            root_cause_analysis: Root cause analysis results
            defect_characteristics: Defect characteristics
            
        Returns:
            List of corrective actions
        """
        corrective_actions = []
        category_name = self.root_cause_categories[predicted_category]
        
        if category_name == 'process_parameters':
            corrective_actions.extend(self._generate_process_parameter_actions(root_cause_analysis))
        elif category_name == 'material_issues':
            corrective_actions.extend(self._generate_material_issue_actions(root_cause_analysis))
        elif category_name == 'equipment_problems':
            corrective_actions.extend(self._generate_equipment_problem_actions(root_cause_analysis))
        elif category_name == 'environmental_factors':
            corrective_actions.extend(self._generate_environmental_factor_actions(root_cause_analysis))
        elif category_name == 'design_issues':
            corrective_actions.extend(self._generate_design_issue_actions(root_cause_analysis))
        elif category_name == 'operator_errors':
            corrective_actions.extend(self._generate_operator_error_actions(root_cause_analysis))
        
        return corrective_actions
    
    def _generate_process_parameter_actions(self, root_cause_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate corrective actions for process parameter issues."""
        actions = []
        
        for factor in root_cause_analysis['specific_factors']:
            if factor['issue'] == 'too_high':
                actions.append({
                    'action': f'Reduce {factor["factor"]}',
                    'priority': 'high',
                    'description': f'Reduce {factor["factor"]} from {factor["value"]} to optimal range',
                    'expected_effect': 'Eliminate defect caused by excessive parameter'
                })
            elif factor['issue'] == 'too_low':
                actions.append({
                    'action': f'Increase {factor["factor"]}',
                    'priority': 'high',
                    'description': f'Increase {factor["factor"]} from {factor["value"]} to optimal range',
                    'expected_effect': 'Eliminate defect caused by insufficient parameter'
                })
        
        return actions
    
    def _generate_material_issue_actions(self, root_cause_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate corrective actions for material issues."""
        actions = []
        
        for factor in root_cause_analysis['specific_factors']:
            if factor['issue'] == 'poor_quality':
                actions.append({
                    'action': 'Improve powder quality',
                    'priority': 'high',
                    'description': 'Source higher quality powder or improve powder processing',
                    'expected_effect': 'Eliminate defects caused by poor powder quality'
                })
            elif factor['issue'] == 'contamination':
                actions.append({
                    'action': 'Eliminate powder contamination',
                    'priority': 'high',
                    'description': 'Improve powder handling and storage procedures',
                    'expected_effect': 'Eliminate contamination-related defects'
                })
            elif factor['issue'] == 'moisture':
                actions.append({
                    'action': 'Reduce powder moisture',
                    'priority': 'medium',
                    'description': 'Improve powder storage conditions and drying procedures',
                    'expected_effect': 'Eliminate moisture-related defects'
                })
        
        return actions
    
    def _generate_equipment_problem_actions(self, root_cause_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate corrective actions for equipment problems."""
        actions = []
        
        for factor in root_cause_analysis['specific_factors']:
            if 'laser' in factor['factor']:
                actions.append({
                    'action': 'Maintain laser system',
                    'priority': 'high',
                    'description': f'Service and calibrate {factor["factor"]} to address {factor["issue"]}',
                    'expected_effect': 'Restore proper laser system performance'
                })
            elif 'recoater' in factor['factor']:
                actions.append({
                    'action': 'Maintain recoater system',
                    'priority': 'high',
                    'description': f'Service and calibrate {factor["factor"]} to address {factor["issue"]}',
                    'expected_effect': 'Restore proper recoater system performance'
                })
            elif 'chamber' in factor['factor']:
                actions.append({
                    'action': 'Maintain chamber system',
                    'priority': 'medium',
                    'description': f'Service and calibrate {factor["factor"]} to address {factor["issue"]}',
                    'expected_effect': 'Restore proper chamber system performance'
                })
        
        return actions
    
    def _generate_environmental_factor_actions(self, root_cause_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate corrective actions for environmental factors."""
        actions = []
        
        for factor in root_cause_analysis['specific_factors']:
            if factor['factor'] == 'temperature':
                actions.append({
                    'action': 'Control environmental temperature',
                    'priority': 'medium',
                    'description': f'Maintain temperature within optimal range to address {factor["issue"]}',
                    'expected_effect': 'Eliminate temperature-related defects'
                })
            elif factor['factor'] == 'humidity':
                actions.append({
                    'action': 'Control environmental humidity',
                    'priority': 'medium',
                    'description': f'Maintain humidity within optimal range to address {factor["issue"]}',
                    'expected_effect': 'Eliminate humidity-related defects'
                })
            elif factor['factor'] == 'vibration':
                actions.append({
                    'action': 'Reduce environmental vibration',
                    'priority': 'high',
                    'description': 'Isolate equipment from vibration sources',
                    'expected_effect': 'Eliminate vibration-related defects'
                })
        
        return actions
    
    def _generate_design_issue_actions(self, root_cause_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate corrective actions for design issues."""
        actions = []
        
        for factor in root_cause_analysis['specific_factors']:
            if 'overhang' in factor['factor']:
                actions.append({
                    'action': 'Modify overhang design',
                    'priority': 'high',
                    'description': 'Reduce overhang angles or add support structures',
                    'expected_effect': 'Eliminate overhang-related defects'
                })
            elif 'wall_thickness' in factor['factor']:
                actions.append({
                    'action': 'Modify wall thickness',
                    'priority': 'high',
                    'description': 'Increase wall thickness or add reinforcement',
                    'expected_effect': 'Eliminate wall thickness-related defects'
                })
            elif 'support' in factor['factor']:
                actions.append({
                    'action': 'Improve support design',
                    'priority': 'medium',
                    'description': 'Add or modify support structures',
                    'expected_effect': 'Eliminate support-related defects'
                })
        
        return actions
    
    def _generate_operator_error_actions(self, root_cause_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate corrective actions for operator errors."""
        actions = []
        
        for factor in root_cause_analysis['specific_factors']:
            if factor['issue'] == 'incorrect':
                actions.append({
                    'action': 'Train operators on correct procedures',
                    'priority': 'high',
                    'description': f'Provide training on correct {factor["factor"]} procedures',
                    'expected_effect': 'Prevent operator-related defects'
                })
            elif factor['issue'] == 'contamination':
                actions.append({
                    'action': 'Improve operator training on contamination prevention',
                    'priority': 'high',
                    'description': 'Train operators on proper powder handling procedures',
                    'expected_effect': 'Prevent contamination-related defects'
                })
            elif factor['issue'] == 'inadequate':
                actions.append({
                    'action': 'Improve operator training on build preparation',
                    'priority': 'medium',
                    'description': 'Train operators on proper build preparation procedures',
                    'expected_effect': 'Prevent preparation-related defects'
                })
        
        return actions
    
    def _calculate_root_cause_impact(self, predicted_category: int, 
                                   defect_characteristics: Dict[str, Any],
                                   process_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate impact of root cause on process and quality.
        
        Args:
            predicted_category: Predicted root cause category
            defect_characteristics: Defect characteristics
            process_data: Process data
            
        Returns:
            Dictionary with impact assessment
        """
        category_name = self.root_cause_categories[predicted_category]
        defect_type = defect_characteristics.get('defect_type', 'unknown')
        defect_size = defect_characteristics.get('defect_size', 0)
        
        # Calculate impact based on category
        if category_name == 'process_parameters':
            impact_level = 'high'
            impact_description = 'Process parameter issues can affect multiple parts and require immediate correction'
        elif category_name == 'equipment_problems':
            impact_level = 'critical'
            impact_description = 'Equipment problems can cause widespread defects and require immediate maintenance'
        elif category_name == 'material_issues':
            impact_level = 'high'
            impact_description = 'Material issues can affect multiple parts and require material replacement'
        elif category_name == 'environmental_factors':
            impact_level = 'medium'
            impact_description = 'Environmental factors can affect part quality and require environmental control'
        elif category_name == 'design_issues':
            impact_level = 'high'
            impact_description = 'Design issues can affect part functionality and require design modification'
        elif category_name == 'operator_errors':
            impact_level = 'medium'
            impact_description = 'Operator errors can affect part quality and require operator training'
        else:
            impact_level = 'unknown'
            impact_description = 'Unknown impact level'
        
        # Adjust impact based on defect characteristics
        if defect_size > 1.0:
            impact_level = 'critical'
        elif defect_type in ['crack', 'lack_of_fusion']:
            impact_level = 'high'
        
        return {
            'impact_level': impact_level,
            'impact_description': impact_description,
            'process_impact': self._assess_process_impact(category_name, defect_type),
            'quality_impact': self._assess_quality_impact(category_name, defect_type),
            'cost_impact': self._assess_cost_impact(category_name, defect_type),
            'timeline_impact': self._assess_timeline_impact(category_name, defect_type)
        }
    
    def _assess_process_impact(self, category_name: str, defect_type: str) -> str:
        """Assess impact on process."""
        if category_name in ['process_parameters', 'equipment_problems']:
            return 'high'
        elif category_name in ['material_issues', 'design_issues']:
            return 'medium'
        else:
            return 'low'
    
    def _assess_quality_impact(self, category_name: str, defect_type: str) -> str:
        """Assess impact on quality."""
        if defect_type in ['crack', 'lack_of_fusion']:
            return 'critical'
        elif defect_type in ['pore', 'contamination']:
            return 'high'
        else:
            return 'medium'
    
    def _assess_cost_impact(self, category_name: str, defect_type: str) -> str:
        """Assess cost impact."""
        if category_name == 'equipment_problems':
            return 'high'
        elif category_name in ['process_parameters', 'material_issues']:
            return 'medium'
        else:
            return 'low'
    
    def _assess_timeline_impact(self, category_name: str, defect_type: str) -> str:
        """Assess timeline impact."""
        if category_name == 'equipment_problems':
            return 'high'
        elif category_name in ['process_parameters', 'material_issues']:
            return 'medium'
        else:
            return 'low'
    
    def _generate_analysis_summary(self, predicted_category: int, confidence: float, 
                                 root_cause_analysis: Dict[str, Any]) -> str:
        """
        Generate analysis summary.
        
        Args:
            predicted_category: Predicted root cause category
            confidence: Analysis confidence
            root_cause_analysis: Root cause analysis results
            
        Returns:
            Analysis summary string
        """
        category_name = self.root_cause_categories[predicted_category]
        severity_level = root_cause_analysis['factor_severity']['severity_level']
        analysis_depth = root_cause_analysis['analysis_depth']
        
        if confidence > 0.8:
            confidence_desc = 'high confidence'
        elif confidence > 0.6:
            confidence_desc = 'medium confidence'
        else:
            confidence_desc = 'low confidence'
        
        return f"Root cause analysis indicates {category_name} as the primary cause with {confidence_desc}. " \
               f"Analysis shows {severity_level} severity factors with {analysis_depth} analysis depth. " \
               f"Immediate corrective actions are recommended to address the identified root causes."
