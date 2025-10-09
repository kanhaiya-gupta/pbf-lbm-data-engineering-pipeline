"""
Defect Detection Real-time Service

This module implements the real-time service for defect detection in PBF-LB/M processes.
It provides REST API endpoints for real-time defect prediction, image defect classification,
defect severity assessment, and root cause analysis.
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import mlflow
import mlflow.tensorflow
from PIL import Image
import io
import base64

from ...pipelines.inference.real_time_inference import RealTimeInferencePipeline
from ...models.defect_detection import RealTimeDefectPredictor, ImageDefectClassifier, DefectSeverityAssessor, RootCauseAnalyzer
from ...config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


# Pydantic models for API requests and responses
class DefectPredictionRequest(BaseModel):
    """Request model for defect prediction."""
    sensor_data: Dict[str, float] = Field(..., description="Sensor data (temperature, vibration, pressure)")
    process_data: Dict[str, Any] = Field(..., description="Process data (laser_power, scan_speed, etc.)")
    material_type: str = Field(..., description="Material type")
    build_parameters: Dict[str, float] = Field(..., description="Build parameters")
    timestamp: Optional[str] = Field(None, description="Data timestamp")


class DefectPredictionResponse(BaseModel):
    """Response model for defect prediction."""
    defect_probability: float = Field(..., description="Defect probability (0-1)")
    defect_type: str = Field(..., description="Predicted defect type")
    severity_level: str = Field(..., description="Severity level (low, medium, high, critical)")
    confidence: float = Field(..., description="Prediction confidence")
    recommendations: List[str] = Field(..., description="Recommendations to prevent defects")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Response timestamp")


class ImageDefectRequest(BaseModel):
    """Request model for image defect classification."""
    image_data: str = Field(..., description="Base64 encoded image data")
    image_type: str = Field(..., description="Image type (ct_scan, powder_bed, surface)")
    material_type: str = Field(..., description="Material type")
    process_parameters: Optional[Dict[str, float]] = Field(None, description="Process parameters")


class ImageDefectResponse(BaseModel):
    """Response model for image defect classification."""
    defect_detected: bool = Field(..., description="Whether defect is detected")
    defect_type: str = Field(..., description="Type of defect detected")
    defect_location: Dict[str, float] = Field(..., description="Defect location coordinates")
    defect_size: float = Field(..., description="Defect size (pixels or mm²)")
    confidence: float = Field(..., description="Classification confidence")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Response timestamp")


class SeverityAssessmentRequest(BaseModel):
    """Request model for defect severity assessment."""
    defect_type: str = Field(..., description="Type of defect")
    defect_size: float = Field(..., description="Defect size")
    defect_location: Dict[str, float] = Field(..., description="Defect location")
    material_type: str = Field(..., description="Material type")
    part_criticality: str = Field(..., description="Part criticality (low, medium, high, critical)")
    process_parameters: Optional[Dict[str, float]] = Field(None, description="Process parameters")


class SeverityAssessmentResponse(BaseModel):
    """Response model for defect severity assessment."""
    severity_score: float = Field(..., description="Severity score (0-1)")
    severity_level: str = Field(..., description="Severity level (low, medium, high, critical)")
    impact_assessment: Dict[str, Any] = Field(..., description="Impact assessment")
    recommended_actions: List[str] = Field(..., description="Recommended actions")
    confidence: float = Field(..., description="Assessment confidence")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Response timestamp")


class RootCauseRequest(BaseModel):
    """Request model for root cause analysis."""
    defect_type: str = Field(..., description="Type of defect")
    defect_details: Dict[str, Any] = Field(..., description="Defect details")
    process_history: List[Dict[str, Any]] = Field(..., description="Process history data")
    material_type: str = Field(..., description="Material type")
    environmental_conditions: Optional[Dict[str, float]] = Field(None, description="Environmental conditions")


class RootCauseResponse(BaseModel):
    """Response model for root cause analysis."""
    root_causes: List[Dict[str, Any]] = Field(..., description="Identified root causes")
    cause_probabilities: List[float] = Field(..., description="Root cause probabilities")
    contributing_factors: List[str] = Field(..., description="Contributing factors")
    prevention_recommendations: List[str] = Field(..., description="Prevention recommendations")
    confidence: float = Field(..., description="Analysis confidence")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Response timestamp")


class DefectDetectionService:
    """
    Real-time defect detection service for PBF-LB/M processes.
    
    This service provides real-time defect detection for:
    - Real-time defect prediction
    - Image defect classification
    - Defect severity assessment
    - Root cause analysis
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the defect detection service.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.app = FastAPI(
            title="Defect Detection Service",
            description="Real-time defect detection for PBF-LB/M manufacturing",
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
            'real_time_defect_predictor': RealTimeDefectPredictor(self.config_manager),
            'image_defect_classifier': ImageDefectClassifier(self.config_manager),
            'defect_severity_assessor': DefectSeverityAssessor(self.config_manager),
            'root_cause_analyzer': RootCauseAnalyzer(self.config_manager)
        }
        
        # Service metrics
        self.service_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'last_request_time': None,
            'defects_detected': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Initialized DefectDetectionService")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": "defect_detection",
                "timestamp": datetime.now().isoformat(),
                "metrics": self.service_metrics
            }
        
        @self.app.post("/predict-defect", response_model=DefectPredictionResponse)
        async def predict_defect(request: DefectPredictionRequest):
            """Predict defects in real-time."""
            return await self._predict_defect(request)
        
        @self.app.post("/classify-image-defect", response_model=ImageDefectResponse)
        async def classify_image_defect(request: ImageDefectRequest):
            """Classify defects in images."""
            return await self._classify_image_defect(request)
        
        @self.app.post("/assess-severity", response_model=SeverityAssessmentResponse)
        async def assess_severity(request: SeverityAssessmentRequest):
            """Assess defect severity."""
            return await self._assess_severity(request)
        
        @self.app.post("/analyze-root-cause", response_model=RootCauseResponse)
        async def analyze_root_cause(request: RootCauseRequest):
            """Analyze root cause of defects."""
            return await self._analyze_root_cause(request)
        
        @self.app.post("/upload-image")
        async def upload_image(file: UploadFile = File(...)):
            """Upload image for defect analysis."""
            return await self._upload_image(file)
        
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
    
    async def _predict_defect(self, request: DefectPredictionRequest) -> DefectPredictionResponse:
        """
        Predict defects in real-time.
        
        Args:
            request: Defect prediction request
            
        Returns:
            Defect prediction response
        """
        start_time = time.time()
        
        try:
            # Prepare input data
            input_data = {
                'timestamp': request.timestamp or datetime.now().isoformat(),
                'sensor_data': request.sensor_data,
                'process_data': {
                    **request.process_data,
                    'material_type': request.material_type,
                    'build_parameters': request.build_parameters
                }
            }
            
            # Get model configurations
            model_configs = [
                {
                    'model_name': 'real_time_defect_predictor',
                    'version': 'latest'
                }
            ]
            
            # Generate predictions
            predictions = await self.inference_pipeline.process_real_time_data(input_data, model_configs)
            
            if predictions['status'] == 'error':
                raise HTTPException(status_code=500, detail=predictions.get('error', 'Prediction failed'))
            
            # Extract defect prediction results
            defect_results = predictions['predictions'].get('real_time_defect_predictor', {})
            
            # Process defect prediction results
            defect_probability = defect_results.get('prediction', [0.1])[0] if defect_results.get('prediction') else 0.1
            confidence = defect_results.get('confidence', [0.8])[0] if defect_results.get('confidence') else 0.8
            
            # Determine defect type and severity
            defect_type = self._determine_defect_type(defect_probability, request.sensor_data, request.process_data)
            severity_level = self._determine_severity_level(defect_probability, defect_type)
            
            # Generate recommendations
            recommendations = self._generate_defect_recommendations(defect_type, severity_level, request)
            
            # Update metrics
            if defect_probability > 0.5:
                self.service_metrics['defects_detected'] += 1
            
            self._update_service_metrics(start_time, True)
            
            return DefectPredictionResponse(
                defect_probability=defect_probability,
                defect_type=defect_type,
                severity_level=severity_level,
                confidence=confidence,
                recommendations=recommendations,
                processing_time=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Defect prediction failed: {e}")
            self._update_service_metrics(start_time, False)
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _classify_image_defect(self, request: ImageDefectRequest) -> ImageDefectResponse:
        """
        Classify defects in images.
        
        Args:
            request: Image defect request
            
        Returns:
            Image defect response
        """
        start_time = time.time()
        
        try:
            # Decode image data
            try:
                image_data = base64.b64decode(request.image_data)
                image = Image.open(io.BytesIO(image_data))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")
            
            # Prepare input data
            input_data = {
                'timestamp': datetime.now().isoformat(),
                'sensor_data': {
                    'image_width': image.width,
                    'image_height': image.height,
                    'image_type': request.image_type
                },
                'process_data': {
                    'material_type': request.material_type,
                    'process_parameters': request.process_parameters or {}
                }
            }
            
            # Get model configurations
            model_configs = [
                {
                    'model_name': 'image_defect_classifier',
                    'version': 'latest'
                }
            ]
            
            # Generate predictions
            predictions = await self.inference_pipeline.process_real_time_data(input_data, model_configs)
            
            if predictions['status'] == 'error':
                raise HTTPException(status_code=500, detail=predictions.get('error', 'Prediction failed'))
            
            # Extract image defect results
            image_results = predictions['predictions'].get('image_defect_classifier', {})
            
            # Process image defect results
            defect_detected = image_results.get('prediction', [0])[0] > 0.5 if image_results.get('prediction') else False
            confidence = image_results.get('confidence', [0.8])[0] if image_results.get('confidence') else 0.8
            
            # Determine defect details
            defect_type = self._determine_image_defect_type(image_results, request.image_type)
            defect_location = self._estimate_defect_location(image_results, image.width, image.height)
            defect_size = self._estimate_defect_size(image_results, image.width, image.height)
            
            # Update metrics
            if defect_detected:
                self.service_metrics['defects_detected'] += 1
            
            self._update_service_metrics(start_time, True)
            
            return ImageDefectResponse(
                defect_detected=defect_detected,
                defect_type=defect_type,
                defect_location=defect_location,
                defect_size=defect_size,
                confidence=confidence,
                processing_time=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Image defect classification failed: {e}")
            self._update_service_metrics(start_time, False)
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _assess_severity(self, request: SeverityAssessmentRequest) -> SeverityAssessmentResponse:
        """
        Assess defect severity.
        
        Args:
            request: Severity assessment request
            
        Returns:
            Severity assessment response
        """
        start_time = time.time()
        
        try:
            # Prepare input data
            input_data = {
                'timestamp': datetime.now().isoformat(),
                'sensor_data': {
                    'defect_size': request.defect_size,
                    'defect_location_x': request.defect_location.get('x', 0),
                    'defect_location_y': request.defect_location.get('y', 0),
                    'defect_location_z': request.defect_location.get('z', 0)
                },
                'process_data': {
                    'defect_type': request.defect_type,
                    'material_type': request.material_type,
                    'part_criticality': request.part_criticality,
                    'process_parameters': request.process_parameters or {}
                }
            }
            
            # Get model configurations
            model_configs = [
                {
                    'model_name': 'defect_severity_assessor',
                    'version': 'latest'
                }
            ]
            
            # Generate predictions
            predictions = await self.inference_pipeline.process_real_time_data(input_data, model_configs)
            
            if predictions['status'] == 'error':
                raise HTTPException(status_code=500, detail=predictions.get('error', 'Prediction failed'))
            
            # Extract severity assessment results
            severity_results = predictions['predictions'].get('defect_severity_assessor', {})
            
            # Process severity assessment results
            severity_score = severity_results.get('prediction', [0.3])[0] if severity_results.get('prediction') else 0.3
            confidence = severity_results.get('confidence', [0.8])[0] if severity_results.get('confidence') else 0.8
            
            # Determine severity level
            severity_level = self._determine_severity_level(severity_score, request.defect_type)
            
            # Generate impact assessment
            impact_assessment = self._generate_impact_assessment(severity_score, request)
            
            # Generate recommended actions
            recommended_actions = self._generate_severity_actions(severity_level, request)
            
            self._update_service_metrics(start_time, True)
            
            return SeverityAssessmentResponse(
                severity_score=severity_score,
                severity_level=severity_level,
                impact_assessment=impact_assessment,
                recommended_actions=recommended_actions,
                confidence=confidence,
                processing_time=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Severity assessment failed: {e}")
            self._update_service_metrics(start_time, False)
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _analyze_root_cause(self, request: RootCauseRequest) -> RootCauseResponse:
        """
        Analyze root cause of defects.
        
        Args:
            request: Root cause request
            
        Returns:
            Root cause response
        """
        start_time = time.time()
        
        try:
            # Prepare input data
            input_data = {
                'timestamp': datetime.now().isoformat(),
                'sensor_data': {
                    'defect_type': request.defect_type,
                    'environmental_temperature': request.environmental_conditions.get('temperature', 20) if request.environmental_conditions else 20,
                    'environmental_humidity': request.environmental_conditions.get('humidity', 50) if request.environmental_conditions else 50
                },
                'process_data': {
                    'defect_details': request.defect_details,
                    'process_history': request.process_history,
                    'material_type': request.material_type
                }
            }
            
            # Get model configurations
            model_configs = [
                {
                    'model_name': 'root_cause_analyzer',
                    'version': 'latest'
                }
            ]
            
            # Generate predictions
            predictions = await self.inference_pipeline.process_real_time_data(input_data, model_configs)
            
            if predictions['status'] == 'error':
                raise HTTPException(status_code=500, detail=predictions.get('error', 'Prediction failed'))
            
            # Extract root cause results
            root_cause_results = predictions['predictions'].get('root_cause_analyzer', {})
            
            # Process root cause results
            root_causes = self._extract_root_causes(root_cause_results, request)
            cause_probabilities = root_cause_results.get('prediction', [0.3, 0.2, 0.1]) if root_cause_results.get('prediction') else [0.3, 0.2, 0.1]
            confidence = root_cause_results.get('confidence', [0.8])[0] if root_cause_results.get('confidence') else 0.8
            
            # Generate contributing factors
            contributing_factors = self._identify_contributing_factors(root_causes, request)
            
            # Generate prevention recommendations
            prevention_recommendations = self._generate_prevention_recommendations(root_causes, request)
            
            self._update_service_metrics(start_time, True)
            
            return RootCauseResponse(
                root_causes=root_causes,
                cause_probabilities=cause_probabilities,
                contributing_factors=contributing_factors,
                prevention_recommendations=prevention_recommendations,
                confidence=confidence,
                processing_time=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Root cause analysis failed: {e}")
            self._update_service_metrics(start_time, False)
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _upload_image(self, file: UploadFile) -> Dict[str, Any]:
        """
        Upload image for defect analysis.
        
        Args:
            file: Uploaded image file
            
        Returns:
            Upload response
        """
        try:
            # Read image data
            image_data = await file.read()
            
            # Convert to base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            return {
                "filename": file.filename,
                "size": len(image_data),
                "image_data": image_base64,
                "upload_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Image upload failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def _determine_defect_type(self, defect_probability: float, sensor_data: Dict[str, float], process_data: Dict[str, Any]) -> str:
        """Determine defect type based on probability and data."""
        if defect_probability < 0.3:
            return "none"
        elif defect_probability < 0.5:
            return "minor_porosity"
        elif defect_probability < 0.7:
            return "cracking"
        elif defect_probability < 0.9:
            return "delamination"
        else:
            return "critical_failure"
    
    def _determine_severity_level(self, score: float, defect_type: str = None) -> str:
        """Determine severity level based on score."""
        if score < 0.25:
            return "low"
        elif score < 0.5:
            return "medium"
        elif score < 0.75:
            return "high"
        else:
            return "critical"
    
    def _determine_image_defect_type(self, image_results: Dict[str, Any], image_type: str) -> str:
        """Determine image defect type."""
        if image_type == "ct_scan":
            return "internal_porosity"
        elif image_type == "powder_bed":
            return "powder_contamination"
        elif image_type == "surface":
            return "surface_roughness"
        else:
            return "unknown_defect"
    
    def _estimate_defect_location(self, image_results: Dict[str, Any], width: int, height: int) -> Dict[str, float]:
        """Estimate defect location in image."""
        # Mock location estimation
        return {
            "x": width * 0.5,
            "y": height * 0.5,
            "confidence": 0.8
        }
    
    def _estimate_defect_size(self, image_results: Dict[str, Any], width: int, height: int) -> float:
        """Estimate defect size."""
        # Mock size estimation
        return (width * height) * 0.01  # 1% of image area
    
    def _generate_defect_recommendations(self, defect_type: str, severity_level: str, request: DefectPredictionRequest) -> List[str]:
        """Generate defect prevention recommendations."""
        recommendations = []
        
        if defect_type == "cracking":
            recommendations.append("Reduce laser power to prevent thermal stress")
            recommendations.append("Increase preheating temperature")
        elif defect_type == "porosity":
            recommendations.append("Check powder quality and moisture content")
            recommendations.append("Optimize gas flow parameters")
        elif defect_type == "delamination":
            recommendations.append("Reduce layer thickness")
            recommendations.append("Increase laser power for better fusion")
        
        if severity_level in ["high", "critical"]:
            recommendations.append("Stop build process immediately")
            recommendations.append("Inspect equipment for malfunctions")
        
        return recommendations
    
    def _generate_impact_assessment(self, severity_score: float, request: SeverityAssessmentRequest) -> Dict[str, Any]:
        """Generate impact assessment."""
        return {
            "structural_integrity": "compromised" if severity_score > 0.5 else "acceptable",
            "functional_impact": "high" if severity_score > 0.7 else "low",
            "repair_required": severity_score > 0.3,
            "cost_impact": "high" if severity_score > 0.6 else "low"
        }
    
    def _generate_severity_actions(self, severity_level: str, request: SeverityAssessmentRequest) -> List[str]:
        """Generate recommended actions based on severity."""
        actions = []
        
        if severity_level == "critical":
            actions.extend([
                "Immediate build termination",
                "Full equipment inspection",
                "Material quality verification"
            ])
        elif severity_level == "high":
            actions.extend([
                "Reduce build speed",
                "Increase monitoring frequency",
                "Prepare for potential rework"
            ])
        elif severity_level == "medium":
            actions.extend([
                "Adjust process parameters",
                "Monitor closely",
                "Document for analysis"
            ])
        else:
            actions.extend([
                "Continue with monitoring",
                "Log for trend analysis"
            ])
        
        return actions
    
    def _extract_root_causes(self, root_cause_results: Dict[str, Any], request: RootCauseRequest) -> List[Dict[str, Any]]:
        """Extract root causes from analysis results."""
        return [
            {
                "cause": "Process parameter deviation",
                "probability": 0.4,
                "description": "Laser power or speed outside optimal range"
            },
            {
                "cause": "Material contamination",
                "probability": 0.3,
                "description": "Powder quality issues or moisture"
            },
            {
                "cause": "Environmental factors",
                "probability": 0.2,
                "description": "Temperature or humidity variations"
            }
        ]
    
    def _identify_contributing_factors(self, root_causes: List[Dict[str, Any]], request: RootCauseRequest) -> List[str]:
        """Identify contributing factors."""
        factors = []
        
        for cause in root_causes:
            if "process" in cause["cause"].lower():
                factors.append("Inconsistent process monitoring")
            elif "material" in cause["cause"].lower():
                factors.append("Inadequate material handling")
            elif "environmental" in cause["cause"].lower():
                factors.append("Poor environmental control")
        
        return factors
    
    def _generate_prevention_recommendations(self, root_causes: List[Dict[str, Any]], request: RootCauseRequest) -> List[str]:
        """Generate prevention recommendations."""
        recommendations = []
        
        for cause in root_causes:
            if "process" in cause["cause"].lower():
                recommendations.append("Implement real-time process monitoring")
                recommendations.append("Establish parameter control limits")
            elif "material" in cause["cause"].lower():
                recommendations.append("Improve material storage conditions")
                recommendations.append("Implement material quality checks")
            elif "environmental" in cause["cause"].lower():
                recommendations.append("Enhance environmental monitoring")
                recommendations.append("Implement climate control systems")
        
        return recommendations
    
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
    
    def run(self, host: str = "0.0.0.0", port: int = 8002):
        """Run the service."""
        logger.info(f"Starting Defect Detection Service on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# Global service instance
service = DefectDetectionService()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    return service.app


if __name__ == "__main__":
    service.run()
