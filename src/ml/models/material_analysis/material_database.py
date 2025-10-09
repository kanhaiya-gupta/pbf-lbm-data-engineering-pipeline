"""
Material Database Model

This module implements a model for material database and knowledge management
for PBF-LB/M manufacturing processes.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Embedding
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import json

from ..base_model import BaseModel

logger = logging.getLogger(__name__)


class MaterialDatabase(BaseModel):
    """
    Model for material database and knowledge management in PBF-LB/M processes.
    
    This model manages:
    - Material property databases
    - Process parameter recommendations
    - Material compatibility analysis
    - Quality standards and specifications
    - Material selection optimization
    - Knowledge base queries and recommendations
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the material database.
        
        Args:
            config_manager: Configuration manager instance
        """
        super().__init__('material_database', config_manager)
        self.model_type = self.model_info.get('algorithm', 'neural_network')
        self.material_categories = [
            'titanium_alloys', 'steel_alloys', 'aluminum_alloys', 'nickel_alloys',
            'cobalt_chrome', 'tungsten', 'copper_alloys', 'ceramics'
        ]
        
        # Initialize material database
        self.material_database = self._initialize_material_database()
        
        logger.info(f"Initialized MaterialDatabase with algorithm: {self.model_type}")
    
    def _initialize_material_database(self) -> Dict[str, Any]:
        """
        Initialize the material database with common PBF-LB/M materials.
        
        Returns:
            Dictionary with material database
        """
        return {
            'titanium_alloys': {
                'Ti-6Al-4V': {
                    'composition': {'Ti': 90, 'Al': 6, 'V': 4},
                    'density': 4.43,
                    'melting_point': 1604,
                    'thermal_conductivity': 7.0,
                    'recommended_power': 300,
                    'recommended_speed': 1000,
                    'applications': ['aerospace', 'medical', 'automotive'],
                    'quality_grade': 'A'
                },
                'Ti-6Al-7Nb': {
                    'composition': {'Ti': 87, 'Al': 6, 'Nb': 7},
                    'density': 4.52,
                    'melting_point': 1610,
                    'thermal_conductivity': 6.8,
                    'recommended_power': 320,
                    'recommended_speed': 950,
                    'applications': ['medical', 'aerospace'],
                    'quality_grade': 'A'
                }
            },
            'steel_alloys': {
                '316L': {
                    'composition': {'Fe': 70, 'Cr': 17, 'Ni': 12, 'Mo': 2.5},
                    'density': 8.0,
                    'melting_point': 1400,
                    'thermal_conductivity': 16.0,
                    'recommended_power': 400,
                    'recommended_speed': 1200,
                    'applications': ['medical', 'chemical', 'food'],
                    'quality_grade': 'A'
                },
                '17-4PH': {
                    'composition': {'Fe': 75, 'Cr': 17, 'Ni': 4, 'Cu': 4},
                    'density': 7.8,
                    'melting_point': 1440,
                    'thermal_conductivity': 15.0,
                    'recommended_power': 450,
                    'recommended_speed': 1100,
                    'applications': ['aerospace', 'marine', 'oil_gas'],
                    'quality_grade': 'B'
                }
            },
            'aluminum_alloys': {
                'AlSi10Mg': {
                    'composition': {'Al': 90, 'Si': 10, 'Mg': 0.5},
                    'density': 2.7,
                    'melting_point': 660,
                    'thermal_conductivity': 150.0,
                    'recommended_power': 200,
                    'recommended_speed': 1500,
                    'applications': ['automotive', 'aerospace', 'electronics'],
                    'quality_grade': 'A'
                },
                'AlSi12': {
                    'composition': {'Al': 88, 'Si': 12},
                    'density': 2.68,
                    'melting_point': 660,
                    'thermal_conductivity': 160.0,
                    'recommended_power': 180,
                    'recommended_speed': 1600,
                    'applications': ['automotive', 'electronics'],
                    'quality_grade': 'B'
                }
            },
            'nickel_alloys': {
                'Inconel 718': {
                    'composition': {'Ni': 53, 'Cr': 19, 'Fe': 18, 'Nb': 5, 'Mo': 3},
                    'density': 8.2,
                    'melting_point': 1260,
                    'thermal_conductivity': 11.0,
                    'recommended_power': 500,
                    'recommended_speed': 800,
                    'applications': ['aerospace', 'power_generation', 'oil_gas'],
                    'quality_grade': 'A'
                },
                'Inconel 625': {
                    'composition': {'Ni': 61, 'Cr': 22, 'Mo': 9, 'Nb': 3.5},
                    'density': 8.4,
                    'melting_point': 1290,
                    'thermal_conductivity': 9.8,
                    'recommended_power': 520,
                    'recommended_speed': 750,
                    'applications': ['aerospace', 'marine', 'chemical'],
                    'quality_grade': 'A'
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
        
        # Output layer (material recommendation score)
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.training_config.get('learning_rate', 0.001)),
            loss='binary_crossentropy',
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
            X_train: Training features (application requirements, constraints)
            y_train: Training targets (material suitability scores)
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
            X: Input features (application requirements)
            
        Returns:
            Predictions array (material suitability scores)
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
            y_test: Test targets (material suitability scores)
            
        Returns:
            Evaluation metrics dictionary
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before evaluation")
            
            # Make predictions
            predictions = self.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
            recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
            f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
            
            # Calculate latency
            start_time = time.time()
            _ = self.predict(X_test[:10])
            latency = (time.time() - start_time) / 10 * 1000
            
            self.evaluation_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
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
    
    def recommend_material(self, application_requirements: Dict[str, Any], 
                          constraints: Dict[str, Any],
                          performance_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recommend materials based on application requirements and constraints.
        
        Args:
            application_requirements: Application requirements (load, temperature, etc.)
            constraints: Constraints (cost, availability, etc.)
            performance_criteria: Performance criteria (strength, corrosion, etc.)
            
        Returns:
            Dictionary with material recommendations
        """
        try:
            # Analyze requirements
            requirement_analysis = self._analyze_requirements(application_requirements, constraints, performance_criteria)
            
            # Find suitable materials
            suitable_materials = self._find_suitable_materials(requirement_analysis)
            
            # Rank materials by suitability
            ranked_materials = self._rank_materials(suitable_materials, requirement_analysis)
            
            # Generate recommendations
            recommendations = self._generate_material_recommendations(ranked_materials, requirement_analysis)
            
            # Calculate compatibility scores
            compatibility_scores = self._calculate_compatibility_scores(ranked_materials, requirement_analysis)
            
            return {
                'requirement_analysis': requirement_analysis,
                'suitable_materials': suitable_materials,
                'ranked_materials': ranked_materials,
                'recommendations': recommendations,
                'compatibility_scores': compatibility_scores,
                'top_recommendation': ranked_materials[0] if ranked_materials else None
            }
            
        except Exception as e:
            logger.error(f"Failed to recommend material: {e}")
            raise
    
    def _analyze_requirements(self, application_requirements: Dict[str, Any], 
                            constraints: Dict[str, Any],
                            performance_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze application requirements and constraints.
        
        Args:
            application_requirements: Application requirements
            constraints: Constraints
            performance_criteria: Performance criteria
            
        Returns:
            Dictionary with requirement analysis
        """
        analysis = {
            'application_type': application_requirements.get('application_type', 'general'),
            'load_requirements': {
                'tensile_strength_min': application_requirements.get('tensile_strength_min', 0),
                'yield_strength_min': application_requirements.get('yield_strength_min', 0),
                'fatigue_strength_min': application_requirements.get('fatigue_strength_min', 0)
            },
            'environmental_requirements': {
                'temperature_range': application_requirements.get('temperature_range', [0, 100]),
                'corrosion_resistance': application_requirements.get('corrosion_resistance', 'low'),
                'oxidation_resistance': application_requirements.get('oxidation_resistance', 'low')
            },
            'constraints': {
                'cost_limit': constraints.get('cost_limit', float('inf')),
                'availability': constraints.get('availability', 'high'),
                'processing_difficulty': constraints.get('processing_difficulty', 'medium')
            },
            'performance_priorities': {
                'strength_priority': performance_criteria.get('strength_priority', 0.3),
                'corrosion_priority': performance_criteria.get('corrosion_priority', 0.2),
                'cost_priority': performance_criteria.get('cost_priority', 0.3),
                'processability_priority': performance_criteria.get('processability_priority', 0.2)
            }
        }
        
        return analysis
    
    def _find_suitable_materials(self, requirement_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find materials that meet the requirements.
        
        Args:
            requirement_analysis: Requirement analysis results
            
        Returns:
            List of suitable materials
        """
        suitable_materials = []
        
        for category, materials in self.material_database.items():
            for material_name, material_data in materials.items():
                # Check if material meets requirements
                if self._check_material_suitability(material_data, requirement_analysis):
                    suitable_materials.append({
                        'name': material_name,
                        'category': category,
                        'data': material_data,
                        'suitability_score': self._calculate_suitability_score(material_data, requirement_analysis)
                    })
        
        return suitable_materials
    
    def _check_material_suitability(self, material_data: Dict[str, Any], 
                                  requirement_analysis: Dict[str, Any]) -> bool:
        """
        Check if material meets basic requirements.
        
        Args:
            material_data: Material data
            requirement_analysis: Requirement analysis
            
        Returns:
            True if material is suitable
        """
        # Check strength requirements
        load_req = requirement_analysis['load_requirements']
        if 'tensile_strength' in material_data:
            if material_data['tensile_strength'] < load_req['tensile_strength_min']:
                return False
        
        # Check environmental requirements
        env_req = requirement_analysis['environmental_requirements']
        temp_range = env_req['temperature_range']
        if 'melting_point' in material_data:
            if material_data['melting_point'] < temp_range[1] + 200:  # Safety margin
                return False
        
        # Check constraints
        constraints = requirement_analysis['constraints']
        if 'cost' in material_data:
            if material_data['cost'] > constraints['cost_limit']:
                return False
        
        return True
    
    def _calculate_suitability_score(self, material_data: Dict[str, Any], 
                                   requirement_analysis: Dict[str, Any]) -> float:
        """
        Calculate suitability score for a material.
        
        Args:
            material_data: Material data
            requirement_analysis: Requirement analysis
            
        Returns:
            Suitability score (0-1)
        """
        score = 0.0
        priorities = requirement_analysis['performance_priorities']
        
        # Strength score
        if 'tensile_strength' in material_data:
            strength_score = min(1.0, material_data['tensile_strength'] / 1000)  # Normalize to 1000 MPa
            score += strength_score * priorities['strength_priority']
        
        # Corrosion resistance score
        if 'corrosion_resistance' in material_data:
            corrosion_score = material_data['corrosion_resistance']
            score += corrosion_score * priorities['corrosion_priority']
        
        # Cost score (lower cost is better)
        if 'cost' in material_data:
            cost_score = max(0.0, 1.0 - material_data['cost'] / 1000)  # Normalize to 1000
            score += cost_score * priorities['cost_priority']
        
        # Processability score
        if 'quality_grade' in material_data:
            grade_scores = {'A': 1.0, 'B': 0.8, 'C': 0.6, 'D': 0.4}
            processability_score = grade_scores.get(material_data['quality_grade'], 0.5)
            score += processability_score * priorities['processability_priority']
        
        return min(1.0, score)
    
    def _rank_materials(self, suitable_materials: List[Dict[str, Any]], 
                       requirement_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rank materials by suitability.
        
        Args:
            suitable_materials: List of suitable materials
            requirement_analysis: Requirement analysis
            
        Returns:
            Ranked list of materials
        """
        # Sort by suitability score
        ranked = sorted(suitable_materials, key=lambda x: x['suitability_score'], reverse=True)
        
        # Add ranking information
        for i, material in enumerate(ranked):
            material['rank'] = i + 1
            material['confidence'] = self._calculate_confidence(material, requirement_analysis)
        
        return ranked
    
    def _calculate_confidence(self, material: Dict[str, Any], 
                            requirement_analysis: Dict[str, Any]) -> float:
        """
        Calculate confidence in material recommendation.
        
        Args:
            material: Material information
            requirement_analysis: Requirement analysis
            
        Returns:
            Confidence score (0-1)
        """
        # Base confidence on suitability score
        base_confidence = material['suitability_score']
        
        # Adjust based on data completeness
        data_completeness = len(material['data']) / 10  # Assume 10 is complete
        completeness_factor = min(1.0, data_completeness)
        
        # Adjust based on quality grade
        quality_grade = material['data'].get('quality_grade', 'C')
        grade_scores = {'A': 1.0, 'B': 0.9, 'C': 0.8, 'D': 0.7}
        quality_factor = grade_scores.get(quality_grade, 0.8)
        
        confidence = base_confidence * completeness_factor * quality_factor
        
        return min(1.0, confidence)
    
    def _generate_material_recommendations(self, ranked_materials: List[Dict[str, Any]], 
                                         requirement_analysis: Dict[str, Any]) -> List[str]:
        """
        Generate material recommendations.
        
        Args:
            ranked_materials: Ranked materials
            requirement_analysis: Requirement analysis
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if not ranked_materials:
            recommendations.append("No suitable materials found for the given requirements")
            recommendations.append("Consider relaxing constraints or requirements")
            return recommendations
        
        # Top recommendation
        top_material = ranked_materials[0]
        recommendations.append(f"Top recommendation: {top_material['name']} (Score: {top_material['suitability_score']:.3f})")
        recommendations.append(f"Category: {top_material['category']}")
        recommendations.append(f"Quality Grade: {top_material['data'].get('quality_grade', 'Unknown')}")
        
        # Alternative recommendations
        if len(ranked_materials) > 1:
            recommendations.append("Alternative materials:")
            for i, material in enumerate(ranked_materials[1:3]):  # Show top 3 alternatives
                recommendations.append(f"{i+2}. {material['name']} (Score: {material['suitability_score']:.3f})")
        
        # Specific recommendations based on requirements
        app_type = requirement_analysis['application_type']
        if app_type == 'aerospace':
            recommendations.append("For aerospace applications, consider high-strength alloys with good fatigue resistance")
        elif app_type == 'medical':
            recommendations.append("For medical applications, prioritize biocompatible materials with excellent corrosion resistance")
        elif app_type == 'automotive':
            recommendations.append("For automotive applications, consider cost-effective materials with good strength-to-weight ratio")
        
        return recommendations
    
    def _calculate_compatibility_scores(self, ranked_materials: List[Dict[str, Any]], 
                                      requirement_analysis: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate compatibility scores for materials.
        
        Args:
            ranked_materials: Ranked materials
            requirement_analysis: Requirement analysis
            
        Returns:
            Dictionary with compatibility scores
        """
        compatibility_scores = {}
        
        for material in ranked_materials:
            material_name = material['name']
            material_data = material['data']
            
            # Calculate compatibility for different aspects
            strength_compatibility = self._calculate_strength_compatibility(material_data, requirement_analysis)
            environmental_compatibility = self._calculate_environmental_compatibility(material_data, requirement_analysis)
            cost_compatibility = self._calculate_cost_compatibility(material_data, requirement_analysis)
            processability_compatibility = self._calculate_processability_compatibility(material_data, requirement_analysis)
            
            compatibility_scores[material_name] = {
                'strength_compatibility': strength_compatibility,
                'environmental_compatibility': environmental_compatibility,
                'cost_compatibility': cost_compatibility,
                'processability_compatibility': processability_compatibility,
                'overall_compatibility': (strength_compatibility + environmental_compatibility + 
                                        cost_compatibility + processability_compatibility) / 4
            }
        
        return compatibility_scores
    
    def _calculate_strength_compatibility(self, material_data: Dict[str, Any], 
                                        requirement_analysis: Dict[str, Any]) -> float:
        """Calculate strength compatibility score."""
        load_req = requirement_analysis['load_requirements']
        
        if 'tensile_strength' in material_data:
            tensile_strength = material_data['tensile_strength']
            min_required = load_req['tensile_strength_min']
            
            if tensile_strength >= min_required:
                return min(1.0, tensile_strength / (min_required * 1.5))  # Bonus for exceeding requirements
            else:
                return max(0.0, tensile_strength / min_required)
        
        return 0.5  # Neutral if no data
    
    def _calculate_environmental_compatibility(self, material_data: Dict[str, Any], 
                                             requirement_analysis: Dict[str, Any]) -> float:
        """Calculate environmental compatibility score."""
        env_req = requirement_analysis['environmental_requirements']
        
        # Temperature compatibility
        temp_range = env_req['temperature_range']
        if 'melting_point' in material_data:
            melting_point = material_data['melting_point']
            max_temp = temp_range[1]
            
            if melting_point > max_temp + 200:  # Safety margin
                return 1.0
            elif melting_point > max_temp:
                return 0.8
            else:
                return 0.3
        
        return 0.5  # Neutral if no data
    
    def _calculate_cost_compatibility(self, material_data: Dict[str, Any], 
                                    requirement_analysis: Dict[str, Any]) -> float:
        """Calculate cost compatibility score."""
        constraints = requirement_analysis['constraints']
        cost_limit = constraints['cost_limit']
        
        if 'cost' in material_data:
            material_cost = material_data['cost']
            
            if material_cost <= cost_limit:
                return 1.0
            else:
                return max(0.0, 1.0 - (material_cost - cost_limit) / cost_limit)
        
        return 0.5  # Neutral if no data
    
    def _calculate_processability_compatibility(self, material_data: Dict[str, Any], 
                                              requirement_analysis: Dict[str, Any]) -> float:
        """Calculate processability compatibility score."""
        if 'quality_grade' in material_data:
            grade = material_data['quality_grade']
            grade_scores = {'A': 1.0, 'B': 0.8, 'C': 0.6, 'D': 0.4}
            return grade_scores.get(grade, 0.5)
        
        return 0.5  # Neutral if no data
    
    def get_material_properties(self, material_name: str) -> Optional[Dict[str, Any]]:
        """
        Get material properties from database.
        
        Args:
            material_name: Name of the material
            
        Returns:
            Material properties or None if not found
        """
        for category, materials in self.material_database.items():
            if material_name in materials:
                return materials[material_name]
        
        return None
    
    def add_material(self, material_name: str, category: str, material_data: Dict[str, Any]) -> bool:
        """
        Add new material to database.
        
        Args:
            material_name: Name of the material
            category: Material category
            material_data: Material data
            
        Returns:
            True if added successfully
        """
        try:
            if category not in self.material_database:
                self.material_database[category] = {}
            
            self.material_database[category][material_name] = material_data
            logger.info(f"Added material {material_name} to category {category}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add material {material_name}: {e}")
            return False
    
    def update_material(self, material_name: str, category: str, material_data: Dict[str, Any]) -> bool:
        """
        Update existing material in database.
        
        Args:
            material_name: Name of the material
            category: Material category
            material_data: Updated material data
            
        Returns:
            True if updated successfully
        """
        try:
            if category in self.material_database and material_name in self.material_database[category]:
                self.material_database[category][material_name].update(material_data)
                logger.info(f"Updated material {material_name} in category {category}")
                return True
            else:
                logger.warning(f"Material {material_name} not found in category {category}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update material {material_name}: {e}")
            return False
    
    def search_materials(self, search_criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search materials based on criteria.
        
        Args:
            search_criteria: Search criteria
            
        Returns:
            List of matching materials
        """
        matching_materials = []
        
        for category, materials in self.material_database.items():
            for material_name, material_data in materials.items():
                if self._matches_criteria(material_data, search_criteria):
                    matching_materials.append({
                        'name': material_name,
                        'category': category,
                        'data': material_data
                    })
        
        return matching_materials
    
    def _matches_criteria(self, material_data: Dict[str, Any], 
                         search_criteria: Dict[str, Any]) -> bool:
        """
        Check if material matches search criteria.
        
        Args:
            material_data: Material data
            search_criteria: Search criteria
            
        Returns:
            True if material matches criteria
        """
        for key, value in search_criteria.items():
            if key in material_data:
                if isinstance(value, tuple) and len(value) == 2:
                    # Range search
                    min_val, max_val = value
                    if not (min_val <= material_data[key] <= max_val):
                        return False
                elif isinstance(value, list):
                    # List search
                    if material_data[key] not in value:
                        return False
                else:
                    # Exact match
                    if material_data[key] != value:
                        return False
        
        return True
