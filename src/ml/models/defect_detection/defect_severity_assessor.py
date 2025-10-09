"""
Defect Severity Assessor Model

This module implements a model for assessing the severity of defects
in PBF-LB/M manufactured parts.
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
import time

from ..base_model import BaseModel

logger = logging.getLogger(__name__)


class DefectSeverityAssessor(BaseModel):
    """
    Model for assessing defect severity in PBF-LB/M manufactured parts.
    
    This model assesses:
    - Defect severity levels (none, low, medium, high, critical)
    - Impact on part functionality
    - Risk assessment for different applications
    - Acceptability for different quality standards
    - Repair recommendations
    - Cost impact assessment
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the defect severity assessor.
        
        Args:
            config_manager: Configuration manager instance
        """
        super().__init__('defect_severity_assessor', config_manager)
        self.model_type = self.model_info.get('algorithm', 'neural_network')
        self.severity_levels = ['none', 'low', 'medium', 'high', 'critical']
        self.quality_standards = ['aerospace', 'medical', 'automotive', 'general']
        
        logger.info(f"Initialized DefectSeverityAssessor with algorithm: {self.model_type}")
    
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
        
        # Output layer (severity score 0-1)
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
            X_train: Training features (defect characteristics, part properties)
            y_train: Training targets (severity scores)
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
            X: Input features (defect characteristics, part properties)
            
        Returns:
            Predictions array (severity scores)
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
            y_test: Test targets (severity scores)
            
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
    
    def assess_defect_severity(self, defect_characteristics: Dict[str, Any], 
                             part_properties: Dict[str, Any],
                             application_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess defect severity for a specific part and application.
        
        Args:
            defect_characteristics: Defect characteristics (type, size, location, etc.)
            part_properties: Part properties (material, geometry, etc.)
            application_requirements: Application requirements (load, environment, etc.)
            
        Returns:
            Dictionary with severity assessment results
        """
        try:
            # Create feature vector from inputs
            features = self._create_feature_vector(defect_characteristics, part_properties, application_requirements)
            
            # Make prediction
            severity_score = self.predict(features.reshape(1, -1))[0]
            
            # Determine severity level
            severity_level = self._score_to_severity_level(severity_score)
            
            # Assess impact on functionality
            functionality_impact = self._assess_functionality_impact(severity_score, defect_characteristics, part_properties)
            
            # Assess risk for different applications
            risk_assessment = self._assess_risk_for_applications(severity_score, application_requirements)
            
            # Determine acceptability for different quality standards
            acceptability_assessment = self._assess_acceptability_for_standards(severity_score, application_requirements)
            
            # Generate repair recommendations
            repair_recommendations = self._generate_repair_recommendations(severity_level, defect_characteristics, part_properties)
            
            # Calculate cost impact
            cost_impact = self._calculate_cost_impact(severity_level, defect_characteristics, part_properties)
            
            return {
                'severity_assessment': {
                    'severity_score': float(severity_score),
                    'severity_level': severity_level,
                    'confidence': self._calculate_confidence(severity_score, defect_characteristics)
                },
                'functionality_impact': functionality_impact,
                'risk_assessment': risk_assessment,
                'acceptability_assessment': acceptability_assessment,
                'repair_recommendations': repair_recommendations,
                'cost_impact': cost_impact,
                'assessment_summary': self._generate_assessment_summary(severity_level, functionality_impact, risk_assessment)
            }
            
        except Exception as e:
            logger.error(f"Failed to assess defect severity: {e}")
            raise
    
    def _create_feature_vector(self, defect_characteristics: Dict[str, Any], 
                              part_properties: Dict[str, Any],
                              application_requirements: Dict[str, Any]) -> np.ndarray:
        """
        Create feature vector from input parameters.
        
        Args:
            defect_characteristics: Defect characteristics
            part_properties: Part properties
            application_requirements: Application requirements
            
        Returns:
            Feature vector
        """
        feature_names = self.get_feature_names()
        features = np.zeros(len(feature_names))
        
        # Combine all input dictionaries
        all_inputs = {**defect_characteristics, **part_properties, **application_requirements}
        
        # Map inputs to features
        for key, value in all_inputs.items():
            if key in feature_names:
                features[feature_names.index(key)] = value
        
        return features
    
    def _score_to_severity_level(self, score: float) -> str:
        """
        Convert severity score to severity level.
        
        Args:
            score: Severity score (0-1)
            
        Returns:
            Severity level string
        """
        if score >= 0.8:
            return 'critical'
        elif score >= 0.6:
            return 'high'
        elif score >= 0.4:
            return 'medium'
        elif score >= 0.2:
            return 'low'
        else:
            return 'none'
    
    def _assess_functionality_impact(self, severity_score: float, 
                                   defect_characteristics: Dict[str, Any],
                                   part_properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess impact on part functionality.
        
        Args:
            severity_score: Severity score
            defect_characteristics: Defect characteristics
            part_properties: Part properties
            
        Returns:
            Dictionary with functionality impact assessment
        """
        defect_type = defect_characteristics.get('defect_type', 'unknown')
        defect_size = defect_characteristics.get('defect_size', 0)
        defect_location = defect_characteristics.get('defect_location', 'unknown')
        
        # Assess impact on different functional aspects
        structural_impact = self._assess_structural_impact(severity_score, defect_type, defect_size, defect_location)
        fatigue_impact = self._assess_fatigue_impact(severity_score, defect_type, defect_size, defect_location)
        corrosion_impact = self._assess_corrosion_impact(severity_score, defect_type, defect_size, defect_location)
        dimensional_impact = self._assess_dimensional_impact(severity_score, defect_type, defect_size, defect_location)
        
        # Calculate overall functionality impact
        overall_impact = (structural_impact + fatigue_impact + corrosion_impact + dimensional_impact) / 4
        
        return {
            'overall_impact': float(overall_impact),
            'structural_impact': float(structural_impact),
            'fatigue_impact': float(fatigue_impact),
            'corrosion_impact': float(corrosion_impact),
            'dimensional_impact': float(dimensional_impact),
            'impact_level': self._score_to_impact_level(overall_impact)
        }
    
    def _assess_structural_impact(self, severity_score: float, defect_type: str, 
                                defect_size: float, defect_location: str) -> float:
        """Assess structural impact of defect."""
        base_impact = severity_score
        
        # Adjust based on defect type
        if defect_type in ['crack', 'lack_of_fusion']:
            base_impact *= 1.5
        elif defect_type in ['pore', 'contamination']:
            base_impact *= 1.2
        
        # Adjust based on defect size
        if defect_size > 1.0:  # Large defect
            base_impact *= 1.3
        elif defect_size < 0.1:  # Small defect
            base_impact *= 0.8
        
        # Adjust based on location
        if defect_location in ['critical_area', 'stress_concentration']:
            base_impact *= 1.4
        elif defect_location in ['surface', 'non_critical']:
            base_impact *= 0.9
        
        return min(1.0, base_impact)
    
    def _assess_fatigue_impact(self, severity_score: float, defect_type: str, 
                             defect_size: float, defect_location: str) -> float:
        """Assess fatigue impact of defect."""
        base_impact = severity_score
        
        # Cracks have high fatigue impact
        if defect_type in ['crack', 'lack_of_fusion']:
            base_impact *= 1.6
        elif defect_type == 'pore':
            base_impact *= 1.3
        
        # Size affects fatigue impact
        if defect_size > 0.5:
            base_impact *= 1.4
        
        # Location affects fatigue impact
        if defect_location in ['critical_area', 'stress_concentration']:
            base_impact *= 1.5
        
        return min(1.0, base_impact)
    
    def _assess_corrosion_impact(self, severity_score: float, defect_type: str, 
                               defect_size: float, defect_location: str) -> float:
        """Assess corrosion impact of defect."""
        base_impact = severity_score
        
        # Surface defects have high corrosion impact
        if defect_type in ['surface_roughness', 'surface_waviness']:
            base_impact *= 1.4
        elif defect_type == 'pore':
            base_impact *= 1.2
        
        # Location affects corrosion impact
        if defect_location == 'surface':
            base_impact *= 1.3
        
        return min(1.0, base_impact)
    
    def _assess_dimensional_impact(self, severity_score: float, defect_type: str, 
                                 defect_size: float, defect_location: str) -> float:
        """Assess dimensional impact of defect."""
        base_impact = severity_score
        
        # Dimensional defects have high impact
        if defect_type in ['overhang', 'warping']:
            base_impact *= 1.5
        elif defect_type in ['surface_roughness', 'surface_waviness']:
            base_impact *= 1.2
        
        return min(1.0, base_impact)
    
    def _score_to_impact_level(self, score: float) -> str:
        """Convert impact score to impact level."""
        if score >= 0.8:
            return 'severe'
        elif score >= 0.6:
            return 'moderate'
        elif score >= 0.4:
            return 'minor'
        else:
            return 'negligible'
    
    def _assess_risk_for_applications(self, severity_score: float, 
                                    application_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess risk for different applications.
        
        Args:
            severity_score: Severity score
            application_requirements: Application requirements
            
        Returns:
            Dictionary with risk assessment for different applications
        """
        application_type = application_requirements.get('application_type', 'general')
        load_level = application_requirements.get('load_level', 'medium')
        environment = application_requirements.get('environment', 'normal')
        
        # Calculate risk for different applications
        aerospace_risk = self._calculate_aerospace_risk(severity_score, load_level, environment)
        medical_risk = self._calculate_medical_risk(severity_score, load_level, environment)
        automotive_risk = self._calculate_automotive_risk(severity_score, load_level, environment)
        general_risk = self._calculate_general_risk(severity_score, load_level, environment)
        
        return {
            'aerospace_risk': float(aerospace_risk),
            'medical_risk': float(medical_risk),
            'automotive_risk': float(automotive_risk),
            'general_risk': float(general_risk),
            'primary_application_risk': self._get_primary_application_risk(application_type, {
                'aerospace': aerospace_risk,
                'medical': medical_risk,
                'automotive': automotive_risk,
                'general': general_risk
            })
        }
    
    def _calculate_aerospace_risk(self, severity_score: float, load_level: str, environment: str) -> float:
        """Calculate risk for aerospace applications."""
        base_risk = severity_score * 1.5  # Aerospace has high safety requirements
        
        if load_level == 'high':
            base_risk *= 1.3
        elif load_level == 'low':
            base_risk *= 0.8
        
        if environment == 'extreme':
            base_risk *= 1.4
        
        return min(1.0, base_risk)
    
    def _calculate_medical_risk(self, severity_score: float, load_level: str, environment: str) -> float:
        """Calculate risk for medical applications."""
        base_risk = severity_score * 1.4  # Medical has high safety requirements
        
        if load_level == 'high':
            base_risk *= 1.2
        
        if environment == 'biocompatible':
            base_risk *= 1.3
        
        return min(1.0, base_risk)
    
    def _calculate_automotive_risk(self, severity_score: float, load_level: str, environment: str) -> float:
        """Calculate risk for automotive applications."""
        base_risk = severity_score * 1.2  # Automotive has moderate safety requirements
        
        if load_level == 'high':
            base_risk *= 1.1
        
        if environment == 'harsh':
            base_risk *= 1.2
        
        return min(1.0, base_risk)
    
    def _calculate_general_risk(self, severity_score: float, load_level: str, environment: str) -> float:
        """Calculate risk for general applications."""
        base_risk = severity_score  # General applications have standard requirements
        
        if load_level == 'high':
            base_risk *= 1.1
        
        return min(1.0, base_risk)
    
    def _get_primary_application_risk(self, application_type: str, risk_dict: Dict[str, float]) -> float:
        """Get risk for primary application type."""
        return risk_dict.get(application_type, risk_dict['general'])
    
    def _assess_acceptability_for_standards(self, severity_score: float, 
                                          application_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess acceptability for different quality standards.
        
        Args:
            severity_score: Severity score
            application_requirements: Application requirements
            
        Returns:
            Dictionary with acceptability assessment
        """
        acceptability = {}
        
        for standard in self.quality_standards:
            if standard == 'aerospace':
                # Aerospace has strict requirements
                acceptable = severity_score < 0.3
                threshold = 0.3
            elif standard == 'medical':
                # Medical has strict requirements
                acceptable = severity_score < 0.4
                threshold = 0.4
            elif standard == 'automotive':
                # Automotive has moderate requirements
                acceptable = severity_score < 0.6
                threshold = 0.6
            else:  # general
                # General has relaxed requirements
                acceptable = severity_score < 0.8
                threshold = 0.8
            
            acceptability[standard] = {
                'acceptable': acceptable,
                'threshold': threshold,
                'margin': threshold - severity_score,
                'compliance_level': self._calculate_compliance_level(severity_score, threshold)
            }
        
        return acceptability
    
    def _calculate_compliance_level(self, severity_score: float, threshold: float) -> str:
        """Calculate compliance level based on severity score and threshold."""
        if severity_score < threshold * 0.5:
            return 'excellent'
        elif severity_score < threshold * 0.8:
            return 'good'
        elif severity_score < threshold:
            return 'acceptable'
        else:
            return 'non_compliant'
    
    def _generate_repair_recommendations(self, severity_level: str, 
                                       defect_characteristics: Dict[str, Any],
                                       part_properties: Dict[str, Any]) -> List[str]:
        """
        Generate repair recommendations based on severity assessment.
        
        Args:
            severity_level: Severity level
            defect_characteristics: Defect characteristics
            part_properties: Part properties
            
        Returns:
            List of repair recommendations
        """
        recommendations = []
        defect_type = defect_characteristics.get('defect_type', 'unknown')
        
        if severity_level == 'none':
            recommendations.append("No repair required - part meets quality standards")
        
        elif severity_level == 'low':
            recommendations.append("Minor repair may be beneficial:")
            recommendations.append("- Consider post-processing surface treatment")
            recommendations.append("- Evaluate if repair cost is justified")
        
        elif severity_level == 'medium':
            recommendations.append("Repair recommended:")
            if defect_type in ['pore', 'contamination']:
                recommendations.append("- Consider welding or brazing repair")
                recommendations.append("- Evaluate part replacement vs repair")
            elif defect_type in ['surface_roughness', 'surface_waviness']:
                recommendations.append("- Consider machining or polishing")
                recommendations.append("- Evaluate surface treatment options")
        
        elif severity_level == 'high':
            recommendations.append("Repair strongly recommended:")
            recommendations.append("- Evaluate repair feasibility and cost")
            recommendations.append("- Consider part replacement if repair is not feasible")
            if defect_type in ['crack', 'lack_of_fusion']:
                recommendations.append("- Structural repair required")
                recommendations.append("- Consider heat treatment after repair")
        
        elif severity_level == 'critical':
            recommendations.append("CRITICAL: Immediate action required:")
            recommendations.append("- Part should be rejected or replaced")
            recommendations.append("- Do not use in service without repair")
            recommendations.append("- Investigate root cause to prevent recurrence")
        
        # Add defect-specific recommendations
        if defect_type == 'crack':
            recommendations.append("Crack repair considerations:")
            recommendations.append("- Ensure complete crack removal")
            recommendations.append("- Consider stress relief after repair")
            recommendations.append("- Verify repair quality with NDT")
        
        elif defect_type == 'pore':
            recommendations.append("Porosity repair considerations:")
            recommendations.append("- Evaluate pore size and distribution")
            recommendations.append("- Consider impregnation or welding")
            recommendations.append("- Verify repair effectiveness")
        
        return recommendations
    
    def _calculate_cost_impact(self, severity_level: str, 
                             defect_characteristics: Dict[str, Any],
                             part_properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate cost impact of defect.
        
        Args:
            severity_level: Severity level
            defect_characteristics: Defect characteristics
            part_properties: Part properties
            
        Returns:
            Dictionary with cost impact assessment
        """
        # Base cost impact by severity level
        base_costs = {
            'none': 0,
            'low': 100,
            'medium': 500,
            'high': 2000,
            'critical': 5000
        }
        
        base_cost = base_costs.get(severity_level, 0)
        
        # Adjust based on defect characteristics
        defect_type = defect_characteristics.get('defect_type', 'unknown')
        defect_size = defect_characteristics.get('defect_size', 0)
        
        if defect_type in ['crack', 'lack_of_fusion']:
            base_cost *= 1.5  # Structural defects are more expensive to repair
        elif defect_type in ['pore', 'contamination']:
            base_cost *= 1.2
        
        if defect_size > 1.0:
            base_cost *= 1.3  # Large defects are more expensive to repair
        
        # Adjust based on part properties
        part_complexity = part_properties.get('complexity', 'medium')
        if part_complexity == 'high':
            base_cost *= 1.4
        elif part_complexity == 'low':
            base_cost *= 0.8
        
        return {
            'repair_cost': float(base_cost),
            'replacement_cost': float(base_cost * 3),  # Replacement is typically 3x repair cost
            'cost_impact_level': self._get_cost_impact_level(base_cost),
            'cost_justification': self._get_cost_justification(severity_level, base_cost)
        }
    
    def _get_cost_impact_level(self, cost: float) -> str:
        """Get cost impact level based on cost."""
        if cost == 0:
            return 'none'
        elif cost < 200:
            return 'low'
        elif cost < 1000:
            return 'medium'
        elif cost < 3000:
            return 'high'
        else:
            return 'critical'
    
    def _get_cost_justification(self, severity_level: str, cost: float) -> str:
        """Get cost justification based on severity and cost."""
        if severity_level in ['none', 'low']:
            return 'Repair cost may not be justified - evaluate cost-benefit'
        elif severity_level == 'medium':
            return 'Repair cost is justified for quality improvement'
        elif severity_level == 'high':
            return 'Repair cost is justified for safety and functionality'
        else:  # critical
            return 'Repair cost is justified - part cannot be used without repair'
    
    def _calculate_confidence(self, severity_score: float, 
                            defect_characteristics: Dict[str, Any]) -> float:
        """
        Calculate confidence in severity assessment.
        
        Args:
            severity_score: Severity score
            defect_characteristics: Defect characteristics
            
        Returns:
            Confidence score (0-1)
        """
        # Base confidence on severity score (extreme scores are less confident)
        base_confidence = 1.0 - abs(severity_score - 0.5) * 2
        
        # Adjust based on defect characteristics completeness
        required_fields = ['defect_type', 'defect_size', 'defect_location']
        completeness = sum(1 for field in required_fields if field in defect_characteristics) / len(required_fields)
        
        # Adjust based on defect type clarity
        defect_type = defect_characteristics.get('defect_type', 'unknown')
        if defect_type == 'unknown':
            type_confidence = 0.5
        else:
            type_confidence = 1.0
        
        confidence = (base_confidence + completeness + type_confidence) / 3
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _generate_assessment_summary(self, severity_level: str, 
                                   functionality_impact: Dict[str, Any],
                                   risk_assessment: Dict[str, Any]) -> str:
        """
        Generate assessment summary.
        
        Args:
            severity_level: Severity level
            functionality_impact: Functionality impact assessment
            risk_assessment: Risk assessment
            
        Returns:
            Assessment summary string
        """
        impact_level = functionality_impact['impact_level']
        primary_risk = risk_assessment['primary_application_risk']
        
        if severity_level == 'none':
            return "No defects detected - part meets quality standards and is suitable for use"
        elif severity_level == 'low':
            return f"Minor defects detected with {impact_level} impact - acceptable for most applications"
        elif severity_level == 'medium':
            return f"Moderate defects detected with {impact_level} impact - repair recommended for optimal performance"
        elif severity_level == 'high':
            return f"Significant defects detected with {impact_level} impact - repair strongly recommended"
        else:  # critical
            return f"Critical defects detected with {impact_level} impact - immediate repair or replacement required"
