"""
Quality Assessment Real-time Service

This module implements the real-time service for quality assessment in PBF-LB/M processes.
It provides REST API endpoints for real-time quality score prediction, dimensional accuracy assessment,
surface finish evaluation, and mechanical property prediction.
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import mlflow
import mlflow.tensorflow

from ...pipelines.inference.real_time_inference import RealTimeInferencePipeline
from ...models.quality_assessment import QualityScorePredictor, DimensionalAccuracyPredictor, SurfaceFinishPredictor, MechanicalPropertyPredictor
from ...config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


# Pydantic models for API requests and responses
class QualityAssessmentRequest(BaseModel):
    """Request model for quality assessment."""
    process_parameters: Dict[str, float] = Field(..., description="Process parameters (laser_power, scan_speed, etc.)")
    material_properties: Dict[str, float] = Field(..., description="Material properties")
    build_conditions: Dict[str, Any] = Field(..., description="Build conditions")
    measurement_data: Optional[Dict[str, float]] = Field(None, description="Measurement data")
    timestamp: Optional[str] = Field(None, description="Assessment timestamp")


class QualityAssessmentResponse(BaseModel):
    """Response model for quality assessment."""
    overall_quality_score: float = Field(..., description="Overall quality score (0-1)")
    dimensional_accuracy: float = Field(..., description="Dimensional accuracy score (0-1)")
    surface_finish: float = Field(..., description="Surface finish score (0-1)")
    mechanical_properties: float = Field(..., description="Mechanical properties score (0-1)")
    quality_grade: str = Field(..., description="Quality grade (A, B, C, D, F)")
    confidence: float = Field(..., description="Assessment confidence")
    recommendations: List[str] = Field(..., description="Quality improvement recommendations")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Response timestamp")


class DimensionalAccuracyRequest(BaseModel):
    """Request model for dimensional accuracy assessment."""
    target_dimensions: Dict[str, float] = Field(..., description="Target dimensions (length, width, height)")
    measured_dimensions: Dict[str, float] = Field(..., description="Measured dimensions")
    tolerance_limits: Dict[str, float] = Field(..., description="Tolerance limits")
    material_type: str = Field(..., description="Material type")
    process_parameters: Optional[Dict[str, float]] = Field(None, description="Process parameters")


class DimensionalAccuracyResponse(BaseModel):
    """Response model for dimensional accuracy assessment."""
    accuracy_score: float = Field(..., description="Dimensional accuracy score (0-1)")
    dimensional_errors: Dict[str, float] = Field(..., description="Dimensional errors for each axis")
    tolerance_compliance: Dict[str, bool] = Field(..., description="Tolerance compliance for each dimension")
    accuracy_grade: str = Field(..., description="Accuracy grade (A, B, C, D, F)")
    confidence: float = Field(..., description="Assessment confidence")
    improvement_suggestions: List[str] = Field(..., description="Improvement suggestions")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Response timestamp")


class SurfaceFinishRequest(BaseModel):
    """Request model for surface finish assessment."""
    surface_roughness: float = Field(..., description="Surface roughness (Ra in μm)")
    surface_texture: Dict[str, float] = Field(..., description="Surface texture parameters")
    surface_defects: List[str] = Field(..., description="Surface defects detected")
    material_type: str = Field(..., description="Material type")
    process_parameters: Optional[Dict[str, float]] = Field(None, description="Process parameters")


class SurfaceFinishResponse(BaseModel):
    """Response model for surface finish assessment."""
    finish_score: float = Field(..., description="Surface finish score (0-1)")
    roughness_grade: str = Field(..., description="Roughness grade (Excellent, Good, Fair, Poor)")
    texture_quality: str = Field(..., description="Texture quality assessment")
    defect_impact: float = Field(..., description="Defect impact score (0-1)")
    confidence: float = Field(..., description="Assessment confidence")
    improvement_recommendations: List[str] = Field(..., description="Improvement recommendations")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Response timestamp")


class MechanicalPropertyRequest(BaseModel):
    """Request model for mechanical property prediction."""
    material_type: str = Field(..., description="Material type")
    process_parameters: Dict[str, float] = Field(..., description="Process parameters")
    heat_treatment: Optional[Dict[str, Any]] = Field(None, description="Heat treatment parameters")
    testing_conditions: Optional[Dict[str, float]] = Field(None, description="Testing conditions")


class MechanicalPropertyResponse(BaseModel):
    """Response model for mechanical property prediction."""
    predicted_properties: Dict[str, float] = Field(..., description="Predicted mechanical properties")
    property_scores: Dict[str, float] = Field(..., description="Property scores (0-1)")
    overall_strength_grade: str = Field(..., description="Overall strength grade")
    confidence: float = Field(..., description="Prediction confidence")
    optimization_suggestions: List[str] = Field(..., description="Optimization suggestions")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Response timestamp")


class QualityAssessmentService:
    """
    Real-time quality assessment service for PBF-LB/M processes.
    
    This service provides real-time quality assessment for:
    - Overall quality score prediction
    - Dimensional accuracy assessment
    - Surface finish evaluation
    - Mechanical property prediction
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the quality assessment service.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.app = FastAPI(
            title="Quality Assessment Service",
            description="Real-time quality assessment for PBF-LB/M manufacturing",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize inference pipeline
        self.inference_pipeline = RealTimeInferencePipeline(self.config_manager)
        
        # Initialize models
        self.models = {
            'quality_score_predictor': QualityScorePredictor(self.config_manager),
            'dimensional_accuracy_predictor': DimensionalAccuracyPredictor(self.config_manager),
            'surface_finish_predictor': SurfaceFinishPredictor(self.config_manager),
            'mechanical_property_predictor': MechanicalPropertyPredictor(self.config_manager)
        }
        
        # Service metrics
        self.service_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'last_request_time': None,
            'quality_assessments': 0,
            'high_quality_parts': 0,
            'low_quality_parts': 0
        }
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Initialized QualityAssessmentService")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": "quality_assessment",
                "timestamp": datetime.now().isoformat(),
                "metrics": self.service_metrics
            }
        
        @self.app.post("/assess-quality", response_model=QualityAssessmentResponse)
        async def assess_quality(request: QualityAssessmentRequest):
            """Assess overall quality in real-time."""
            return await self._assess_quality(request)
        
        @self.app.post("/assess-dimensional-accuracy", response_model=DimensionalAccuracyResponse)
        async def assess_dimensional_accuracy(request: DimensionalAccuracyRequest):
            """Assess dimensional accuracy."""
            return await self._assess_dimensional_accuracy(request)
        
        @self.app.post("/assess-surface-finish", response_model=SurfaceFinishResponse)
        async def assess_surface_finish(request: SurfaceFinishRequest):
            """Assess surface finish."""
            return await self._assess_surface_finish(request)
        
        @self.app.post("/predict-mechanical-properties", response_model=MechanicalPropertyResponse)
        async def predict_mechanical_properties(request: MechanicalPropertyRequest):
            """Predict mechanical properties."""
            return await self._predict_mechanical_properties(request)
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get service metrics."""
            return self.service_metrics
        
        @self.app.get("/models")
        async def get_available_models():
            """Get available models."""
            return {
                "available_models": list(self.models.keys()),
                "model_status": {name: "loaded" for name in self.models.keys()}
            }
    
    async def _assess_quality(self, request: QualityAssessmentRequest) -> QualityAssessmentResponse:
        """
        Assess overall quality in real-time.
        
        Args:
            request: Quality assessment request
            
        Returns:
            Quality assessment response
        """
        start_time = time.time()
        
        try:
            # Prepare input data
            input_data = {
                'timestamp': request.timestamp or datetime.now().isoformat(),
                'sensor_data': {
                    'measurement_data': request.measurement_data or {}
                },
                'process_data': {
                    'process_parameters': request.process_parameters,
                    'material_properties': request.material_properties,
                    'build_conditions': request.build_conditions
                }
            }
            
            # Get model configurations
            model_configs = [
                {
                    'model_name': 'quality_score_predictor',
                    'version': 'latest'
                }
            ]
            
            # Generate predictions
            predictions = await self.inference_pipeline.process_real_time_data(input_data, model_configs)
            
            if predictions['status'] == 'error':
                raise HTTPException(status_code=500, detail=predictions.get('error', 'Prediction failed'))
            
            # Extract quality assessment results
            quality_results = predictions['predictions'].get('quality_score_predictor', {})
            
            # Process quality assessment results
            overall_quality_score = quality_results.get('prediction', [0.8])[0] if quality_results.get('prediction') else 0.8
            confidence = quality_results.get('confidence', [0.8])[0] if quality_results.get('confidence') else 0.8
            
            # Calculate individual quality scores
            dimensional_accuracy = self._calculate_dimensional_accuracy_score(request)
            surface_finish = self._calculate_surface_finish_score(request)
            mechanical_properties = self._calculate_mechanical_properties_score(request)
            
            # Determine quality grade
            quality_grade = self._determine_quality_grade(overall_quality_score)
            
            # Generate recommendations
            recommendations = self._generate_quality_recommendations(overall_quality_score, request)
            
            # Update metrics
            self.service_metrics['quality_assessments'] += 1
            if overall_quality_score >= 0.8:
                self.service_metrics['high_quality_parts'] += 1
            else:
                self.service_metrics['low_quality_parts'] += 1
            
            self._update_service_metrics(start_time, True)
            
            return QualityAssessmentResponse(
                overall_quality_score=overall_quality_score,
                dimensional_accuracy=dimensional_accuracy,
                surface_finish=surface_finish,
                mechanical_properties=mechanical_properties,
                quality_grade=quality_grade,
                confidence=confidence,
                recommendations=recommendations,
                processing_time=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            self._update_service_metrics(start_time, False)
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _assess_dimensional_accuracy(self, request: DimensionalAccuracyRequest) -> DimensionalAccuracyResponse:
        """
        Assess dimensional accuracy.
        
        Args:
            request: Dimensional accuracy request
            
        Returns:
            Dimensional accuracy response
        """
        start_time = time.time()
        
        try:
            # Prepare input data
            input_data = {
                'timestamp': datetime.now().isoformat(),
                'sensor_data': {
                    'target_length': request.target_dimensions.get('length', 0),
                    'target_width': request.target_dimensions.get('width', 0),
                    'target_height': request.target_dimensions.get('height', 0),
                    'measured_length': request.measured_dimensions.get('length', 0),
                    'measured_width': request.measured_dimensions.get('width', 0),
                    'measured_height': request.measured_dimensions.get('height', 0)
                },
                'process_data': {
                    'material_type': request.material_type,
                    'tolerance_limits': request.tolerance_limits,
                    'process_parameters': request.process_parameters or {}
                }
            }
            
            # Get model configurations
            model_configs = [
                {
                    'model_name': 'dimensional_accuracy_predictor',
                    'version': 'latest'
                }
            ]
            
            # Generate predictions
            predictions = await self.inference_pipeline.process_real_time_data(input_data, model_configs)
            
            if predictions['status'] == 'error':
                raise HTTPException(status_code=500, detail=predictions.get('error', 'Prediction failed'))
            
            # Extract dimensional accuracy results
            accuracy_results = predictions['predictions'].get('dimensional_accuracy_predictor', {})
            
            # Process dimensional accuracy results
            accuracy_score = accuracy_results.get('prediction', [0.8])[0] if accuracy_results.get('prediction') else 0.8
            confidence = accuracy_results.get('confidence', [0.8])[0] if accuracy_results.get('confidence') else 0.8
            
            # Calculate dimensional errors
            dimensional_errors = self._calculate_dimensional_errors(request)
            
            # Check tolerance compliance
            tolerance_compliance = self._check_tolerance_compliance(request)
            
            # Determine accuracy grade
            accuracy_grade = self._determine_accuracy_grade(accuracy_score)
            
            # Generate improvement suggestions
            improvement_suggestions = self._generate_accuracy_improvements(dimensional_errors, tolerance_compliance)
            
            self._update_service_metrics(start_time, True)
            
            return DimensionalAccuracyResponse(
                accuracy_score=accuracy_score,
                dimensional_errors=dimensional_errors,
                tolerance_compliance=tolerance_compliance,
                accuracy_grade=accuracy_grade,
                confidence=confidence,
                improvement_suggestions=improvement_suggestions,
                processing_time=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Dimensional accuracy assessment failed: {e}")
            self._update_service_metrics(start_time, False)
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _assess_surface_finish(self, request: SurfaceFinishRequest) -> SurfaceFinishResponse:
        """
        Assess surface finish.
        
        Args:
            request: Surface finish request
            
        Returns:
            Surface finish response
        """
        start_time = time.time()
        
        try:
            # Prepare input data
            input_data = {
                'timestamp': datetime.now().isoformat(),
                'sensor_data': {
                    'surface_roughness': request.surface_roughness,
                    'texture_energy': request.surface_texture.get('energy', 0),
                    'texture_contrast': request.surface_texture.get('contrast', 0),
                    'texture_homogeneity': request.surface_texture.get('homogeneity', 0)
                },
                'process_data': {
                    'material_type': request.material_type,
                    'surface_defects': request.surface_defects,
                    'process_parameters': request.process_parameters or {}
                }
            }
            
            # Get model configurations
            model_configs = [
                {
                    'model_name': 'surface_finish_predictor',
                    'version': 'latest'
                }
            ]
            
            # Generate predictions
            predictions = await self.inference_pipeline.process_real_time_data(input_data, model_configs)
            
            if predictions['status'] == 'error':
                raise HTTPException(status_code=500, detail=predictions.get('error', 'Prediction failed'))
            
            # Extract surface finish results
            finish_results = predictions['predictions'].get('surface_finish_predictor', {})
            
            # Process surface finish results
            finish_score = finish_results.get('prediction', [0.8])[0] if finish_results.get('prediction') else 0.8
            confidence = finish_results.get('confidence', [0.8])[0] if finish_results.get('confidence') else 0.8
            
            # Determine roughness grade
            roughness_grade = self._determine_roughness_grade(request.surface_roughness)
            
            # Assess texture quality
            texture_quality = self._assess_texture_quality(request.surface_texture)
            
            # Calculate defect impact
            defect_impact = self._calculate_defect_impact(request.surface_defects)
            
            # Generate improvement recommendations
            improvement_recommendations = self._generate_surface_improvements(request)
            
            self._update_service_metrics(start_time, True)
            
            return SurfaceFinishResponse(
                finish_score=finish_score,
                roughness_grade=roughness_grade,
                texture_quality=texture_quality,
                defect_impact=defect_impact,
                confidence=confidence,
                improvement_recommendations=improvement_recommendations,
                processing_time=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Surface finish assessment failed: {e}")
            self._update_service_metrics(start_time, False)
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _predict_mechanical_properties(self, request: MechanicalPropertyRequest) -> MechanicalPropertyResponse:
        """
        Predict mechanical properties.
        
        Args:
            request: Mechanical property request
            
        Returns:
            Mechanical property response
        """
        start_time = time.time()
        
        try:
            # Prepare input data
            input_data = {
                'timestamp': datetime.now().isoformat(),
                'sensor_data': {
                    'testing_temperature': request.testing_conditions.get('temperature', 20) if request.testing_conditions else 20,
                    'testing_strain_rate': request.testing_conditions.get('strain_rate', 0.001) if request.testing_conditions else 0.001
                },
                'process_data': {
                    'material_type': request.material_type,
                    'process_parameters': request.process_parameters,
                    'heat_treatment': request.heat_treatment or {}
                }
            }
            
            # Get model configurations
            model_configs = [
                {
                    'model_name': 'mechanical_property_predictor',
                    'version': 'latest'
                }
            ]
            
            # Generate predictions
            predictions = await self.inference_pipeline.process_real_time_data(input_data, model_configs)
            
            if predictions['status'] == 'error':
                raise HTTPException(status_code=500, detail=predictions.get('error', 'Prediction failed'))
            
            # Extract mechanical property results
            property_results = predictions['predictions'].get('mechanical_property_predictor', {})
            
            # Process mechanical property results
            predicted_properties = self._extract_predicted_properties(property_results)
            confidence = property_results.get('confidence', [0.8])[0] if property_results.get('confidence') else 0.8
            
            # Calculate property scores
            property_scores = self._calculate_property_scores(predicted_properties, request.material_type)
            
            # Determine overall strength grade
            overall_strength_grade = self._determine_strength_grade(property_scores)
            
            # Generate optimization suggestions
            optimization_suggestions = self._generate_property_optimizations(predicted_properties, request)
            
            self._update_service_metrics(start_time, True)
            
            return MechanicalPropertyResponse(
                predicted_properties=predicted_properties,
                property_scores=property_scores,
                overall_strength_grade=overall_strength_grade,
                confidence=confidence,
                optimization_suggestions=optimization_suggestions,
                processing_time=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Mechanical property prediction failed: {e}")
            self._update_service_metrics(start_time, False)
            raise HTTPException(status_code=500, detail=str(e))
    
    def _calculate_dimensional_accuracy_score(self, request: QualityAssessmentRequest) -> float:
        """Calculate dimensional accuracy score."""
        # Mock calculation based on process parameters
        if 'layer_height' in request.process_parameters:
            layer_height = request.process_parameters['layer_height']
            if layer_height <= 0.05:
                return 0.9
            elif layer_height <= 0.1:
                return 0.8
            else:
                return 0.7
        return 0.8
    
    def _calculate_surface_finish_score(self, request: QualityAssessmentRequest) -> float:
        """Calculate surface finish score."""
        # Mock calculation based on process parameters
        if 'scan_speed' in request.process_parameters:
            scan_speed = request.process_parameters['scan_speed']
            if scan_speed <= 1000:
                return 0.9
            elif scan_speed <= 2000:
                return 0.8
            else:
                return 0.7
        return 0.8
    
    def _calculate_mechanical_properties_score(self, request: QualityAssessmentRequest) -> float:
        """Calculate mechanical properties score."""
        # Mock calculation based on process parameters
        if 'laser_power' in request.process_parameters:
            laser_power = request.process_parameters['laser_power']
            if laser_power >= 200:
                return 0.9
            elif laser_power >= 150:
                return 0.8
            else:
                return 0.7
        return 0.8
    
    def _determine_quality_grade(self, quality_score: float) -> str:
        """Determine quality grade based on score."""
        if quality_score >= 0.9:
            return "A"
        elif quality_score >= 0.8:
            return "B"
        elif quality_score >= 0.7:
            return "C"
        elif quality_score >= 0.6:
            return "D"
        else:
            return "F"
    
    def _determine_accuracy_grade(self, accuracy_score: float) -> str:
        """Determine accuracy grade based on score."""
        if accuracy_score >= 0.95:
            return "A"
        elif accuracy_score >= 0.85:
            return "B"
        elif accuracy_score >= 0.75:
            return "C"
        elif accuracy_score >= 0.65:
            return "D"
        else:
            return "F"
    
    def _determine_roughness_grade(self, roughness: float) -> str:
        """Determine roughness grade based on Ra value."""
        if roughness <= 1.6:
            return "Excellent"
        elif roughness <= 3.2:
            return "Good"
        elif roughness <= 6.3:
            return "Fair"
        else:
            return "Poor"
    
    def _assess_texture_quality(self, surface_texture: Dict[str, float]) -> str:
        """Assess texture quality."""
        energy = surface_texture.get('energy', 0)
        contrast = surface_texture.get('contrast', 0)
        
        if energy > 0.7 and contrast > 0.5:
            return "Excellent"
        elif energy > 0.5 and contrast > 0.3:
            return "Good"
        elif energy > 0.3 and contrast > 0.2:
            return "Fair"
        else:
            return "Poor"
    
    def _calculate_defect_impact(self, surface_defects: List[str]) -> float:
        """Calculate defect impact score."""
        if not surface_defects:
            return 0.0
        
        critical_defects = ['cracks', 'delamination', 'porosity']
        medium_defects = ['scratches', 'discoloration']
        
        impact = 0.0
        for defect in surface_defects:
            if defect in critical_defects:
                impact += 0.3
            elif defect in medium_defects:
                impact += 0.1
            else:
                impact += 0.05
        
        return min(impact, 1.0)
    
    def _extract_predicted_properties(self, property_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract predicted mechanical properties."""
        predictions = property_results.get('prediction', [500, 200, 7.8, 0.3])
        
        return {
            'tensile_strength': predictions[0] if len(predictions) > 0 else 500.0,
            'hardness': predictions[1] if len(predictions) > 1 else 200.0,
            'density': predictions[2] if len(predictions) > 2 else 7.8,
            'elongation': predictions[3] if len(predictions) > 3 else 0.3
        }
    
    def _calculate_property_scores(self, predicted_properties: Dict[str, float], material_type: str) -> Dict[str, float]:
        """Calculate property scores based on material type."""
        # Mock scoring based on material type
        base_scores = {
            'tensile_strength': 0.8,
            'hardness': 0.8,
            'density': 0.8,
            'elongation': 0.8
        }
        
        # Adjust based on material type
        if material_type.lower() == 'titanium':
            base_scores['tensile_strength'] = 0.9
            base_scores['hardness'] = 0.7
        elif material_type.lower() == 'aluminum':
            base_scores['tensile_strength'] = 0.7
            base_scores['hardness'] = 0.6
        
        return base_scores
    
    def _determine_strength_grade(self, property_scores: Dict[str, float]) -> str:
        """Determine overall strength grade."""
        avg_score = sum(property_scores.values()) / len(property_scores)
        
        if avg_score >= 0.9:
            return "Excellent"
        elif avg_score >= 0.8:
            return "Good"
        elif avg_score >= 0.7:
            return "Fair"
        else:
            return "Poor"
    
    def _calculate_dimensional_errors(self, request: DimensionalAccuracyRequest) -> Dict[str, float]:
        """Calculate dimensional errors."""
        errors = {}
        for dimension in ['length', 'width', 'height']:
            if dimension in request.target_dimensions and dimension in request.measured_dimensions:
                target = request.target_dimensions[dimension]
                measured = request.measured_dimensions[dimension]
                errors[dimension] = abs(measured - target)
        return errors
    
    def _check_tolerance_compliance(self, request: DimensionalAccuracyRequest) -> Dict[str, bool]:
        """Check tolerance compliance."""
        compliance = {}
        for dimension in ['length', 'width', 'height']:
            if dimension in request.tolerance_limits:
                tolerance = request.tolerance_limits[dimension]
                if dimension in request.target_dimensions and dimension in request.measured_dimensions:
                    target = request.target_dimensions[dimension]
                    measured = request.measured_dimensions[dimension]
                    error = abs(measured - target)
                    compliance[dimension] = error <= tolerance
        return compliance
    
    def _generate_quality_recommendations(self, quality_score: float, request: QualityAssessmentRequest) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        if quality_score < 0.8:
            recommendations.append("Optimize process parameters for better quality")
            recommendations.append("Check material quality and storage conditions")
            recommendations.append("Verify equipment calibration")
        
        if 'laser_power' in request.process_parameters:
            if request.process_parameters['laser_power'] < 150:
                recommendations.append("Consider increasing laser power for better fusion")
        
        if 'scan_speed' in request.process_parameters:
            if request.process_parameters['scan_speed'] > 2000:
                recommendations.append("Consider reducing scan speed for better surface finish")
        
        return recommendations
    
    def _generate_accuracy_improvements(self, dimensional_errors: Dict[str, float], tolerance_compliance: Dict[str, bool]) -> List[str]:
        """Generate accuracy improvement suggestions."""
        suggestions = []
        
        for dimension, error in dimensional_errors.items():
            if not tolerance_compliance.get(dimension, True):
                suggestions.append(f"Improve {dimension} accuracy - current error: {error:.3f}mm")
        
        if any(not compliant for compliant in tolerance_compliance.values()):
            suggestions.append("Check build orientation and support strategy")
            suggestions.append("Verify layer height and scan parameters")
        
        return suggestions
    
    def _generate_surface_improvements(self, request: SurfaceFinishRequest) -> List[str]:
        """Generate surface finish improvement recommendations."""
        recommendations = []
        
        if request.surface_roughness > 3.2:
            recommendations.append("Reduce scan speed for better surface finish")
            recommendations.append("Optimize laser power and focus")
        
        if request.surface_defects:
            recommendations.append("Check powder quality and contamination")
            recommendations.append("Verify process parameters and environmental conditions")
        
        return recommendations
    
    def _generate_property_optimizations(self, predicted_properties: Dict[str, float], request: MechanicalPropertyRequest) -> List[str]:
        """Generate mechanical property optimization suggestions."""
        suggestions = []
        
        if predicted_properties['tensile_strength'] < 400:
            suggestions.append("Increase laser power for better fusion")
            suggestions.append("Optimize scan strategy for improved strength")
        
        if predicted_properties['hardness'] < 150:
            suggestions.append("Consider post-processing heat treatment")
            suggestions.append("Optimize cooling rate during build")
        
        return suggestions
    
    def _update_service_metrics(self, start_time: float, success: bool):
        """Update service metrics."""
        response_time = time.time() - start_time
        
        self.service_metrics['total_requests'] += 1
        if success:
            self.service_metrics['successful_requests'] += 1
        else:
            self.service_metrics['failed_requests'] += 1
        
        # Update average response time
        total_successful = self.service_metrics['successful_requests']
        if total_successful > 0:
            current_avg = self.service_metrics['average_response_time']
            self.service_metrics['average_response_time'] = (
                (current_avg * (total_successful - 1) + response_time) / total_successful
            )
        
        self.service_metrics['last_request_time'] = datetime.now().isoformat()
    
    def run(self, host: str = "0.0.0.0", port: int = 8003):
        """Run the service."""
        logger.info(f"Starting Quality Assessment Service on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# Global service instance
service = QualityAssessmentService()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    return service.app


if __name__ == "__main__":
    service.run()
