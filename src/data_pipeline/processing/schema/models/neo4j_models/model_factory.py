"""
Neo4j Model Factory

This module provides a factory for creating and managing Neo4j Pydantic models.
"""

from typing import Dict, Any, List, Optional, Type, Union
from enum import Enum
from pydantic import BaseModel, ValidationError as PydanticValidationError
import json
from src.data_pipeline.processing.knowledge_graph.utils.json_parser import safe_json_loads_with_fallback

from .core_graph_models import (
    ProcessModel, MachineModel, PartModel, BuildModel, MaterialModel,
    QualityModel, SensorModel, UserModel, OperatorModel, AlertModel,
    DefectModel, ImageModel, LogModel, InspectionModel,
    # Image Models
    ThermalImageModel, ProcessImageModel, CTScanImageModel, PowderBedImageModel,
    # File Models
    BuildFileModel, ModelFileModel, LogFileModel,
    # Cache Models
    ProcessCacheModel, AnalyticsCacheModel,
    # Queue Models
    JobQueueModel,
    # Session Models
    UserSessionModel,
    # Reading Models
    SensorReadingModel,
    # Event Models
    ProcessMonitoringModel, MachineStatusModel, AlertEventModel,
    # Config Models
    MachineConfigModel,
    # Additional Models
    BatchModel, MeasurementModel, SensorTypeModel
)
from .relationship_models import (
    ProcessMachineRelationship, ProcessPartRelationship, ProcessBuildRelationship,
    ProcessMaterialRelationship, ProcessQualityRelationship, ProcessSensorRelationship,
    ProcessOperatorRelationship, ProcessAlertRelationship, ProcessDefectRelationship,
    ProcessImageRelationship, ProcessLogRelationship
)


class Neo4jModelType(str, Enum):
    """Neo4j model type enumeration."""
    # Core Node Models
    PROCESS = "Process"
    MACHINE = "Machine"
    PART = "Part"
    BUILD = "Build"
    MATERIAL = "Material"
    QUALITY = "Quality"
    SENSOR = "Sensor"
    USER = "User"
    OPERATOR = "Operator"
    ALERT = "Alert"
    DEFECT = "Defect"
    IMAGE = "Image"
    THERMAL_IMAGE = "ThermalImage"
    PROCESS_IMAGE = "ProcessImage"
    CT_SCAN_IMAGE = "CTScanImage"
    POWDER_BED_IMAGE = "PowderBedImage"
    LOG = "Log"
    INSPECTION = "Inspection"
    
    # File Models
    BUILD_FILE = "BuildFile"
    MODEL_FILE = "ModelFile"
    LOG_FILE = "LogFile"
    
    # Cache Models
    PROCESS_CACHE = "ProcessCache"
    ANALYTICS_CACHE = "AnalyticsCache"
    
    # Queue Models
    JOB_QUEUE = "JobQueue"
    
    # Session Models
    USER_SESSION = "UserSession"
    
    # Reading Models
    SENSOR_READING = "SensorReading"
    
    # Event Models
    PROCESS_MONITORING = "ProcessMonitoring"
    MACHINE_STATUS = "MachineStatus"
    ALERT_EVENT = "AlertEvent"
    
    # Config Models
    MACHINE_CONFIG = "MachineConfig"
    
    # Additional Models
    BATCH = "Batch"
    MEASUREMENT = "Measurement"
    SENSOR_TYPE = "SensorType"
    
    # Relationship Models
    USES_MACHINE = "USES_MACHINE"
    CREATES_PART = "CREATES_PART"
    PART_OF_BUILD = "PART_OF_BUILD"
    USES_MATERIAL = "USES_MATERIAL"
    HAS_QUALITY = "HAS_QUALITY"
    MONITORED_BY = "MONITORED_BY"
    OPERATED_BY = "OPERATED_BY"
    GENERATES_ALERT = "GENERATES_ALERT"
    HAS_DEFECT = "HAS_DEFECT"
    CAPTURED_BY = "CAPTURED_BY"
    LOGGED_IN = "LOGGED_IN"


class Neo4jModelFactory:
    """Factory class for Neo4j models."""
    
    # Model registry
    _models: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def _register_default_models(cls):
        """Register default models."""
        # Register node models
        cls.register_model(Neo4jModelType.PROCESS.value, ProcessModel)
        cls.register_model(Neo4jModelType.MACHINE.value, MachineModel)
        cls.register_model(Neo4jModelType.PART.value, PartModel)
        cls.register_model(Neo4jModelType.BUILD.value, BuildModel)
        cls.register_model(Neo4jModelType.MATERIAL.value, MaterialModel)
        cls.register_model(Neo4jModelType.QUALITY.value, QualityModel)
        cls.register_model(Neo4jModelType.SENSOR.value, SensorModel)
        cls.register_model(Neo4jModelType.USER.value, UserModel)
        cls.register_model(Neo4jModelType.OPERATOR.value, OperatorModel)
        cls.register_model(Neo4jModelType.ALERT.value, AlertModel)
        cls.register_model(Neo4jModelType.DEFECT.value, DefectModel)
        cls.register_model(Neo4jModelType.IMAGE.value, ImageModel)
        cls.register_model(Neo4jModelType.THERMAL_IMAGE.value, ThermalImageModel)
        cls.register_model(Neo4jModelType.PROCESS_IMAGE.value, ProcessImageModel)
        cls.register_model(Neo4jModelType.CT_SCAN_IMAGE.value, CTScanImageModel)
        cls.register_model(Neo4jModelType.POWDER_BED_IMAGE.value, PowderBedImageModel)
        cls.register_model(Neo4jModelType.LOG.value, LogModel)
        cls.register_model(Neo4jModelType.INSPECTION.value, InspectionModel)
        
        # Register file models
        cls.register_model(Neo4jModelType.BUILD_FILE.value, BuildFileModel)
        cls.register_model(Neo4jModelType.MODEL_FILE.value, ModelFileModel)
        cls.register_model(Neo4jModelType.LOG_FILE.value, LogFileModel)
        
        # Register cache models
        cls.register_model(Neo4jModelType.PROCESS_CACHE.value, ProcessCacheModel)
        cls.register_model(Neo4jModelType.ANALYTICS_CACHE.value, AnalyticsCacheModel)
        
        # Register queue models
        cls.register_model(Neo4jModelType.JOB_QUEUE.value, JobQueueModel)
        
        # Register session models
        cls.register_model(Neo4jModelType.USER_SESSION.value, UserSessionModel)
        
        # Register reading models
        cls.register_model(Neo4jModelType.SENSOR_READING.value, SensorReadingModel)
        
        # Register event models
        cls.register_model(Neo4jModelType.PROCESS_MONITORING.value, ProcessMonitoringModel)
        cls.register_model(Neo4jModelType.MACHINE_STATUS.value, MachineStatusModel)
        cls.register_model(Neo4jModelType.ALERT_EVENT.value, AlertEventModel)
        
        # Register config models
        cls.register_model(Neo4jModelType.MACHINE_CONFIG.value, MachineConfigModel)
        
        # Register additional models
        cls.register_model(Neo4jModelType.BATCH.value, BatchModel)
        cls.register_model(Neo4jModelType.MEASUREMENT.value, MeasurementModel)
        cls.register_model(Neo4jModelType.SENSOR_TYPE.value, SensorTypeModel)
        
        # Register relationship models
        cls.register_model(Neo4jModelType.USES_MACHINE.value, ProcessMachineRelationship)
        cls.register_model(Neo4jModelType.CREATES_PART.value, ProcessPartRelationship)
        cls.register_model(Neo4jModelType.PART_OF_BUILD.value, ProcessBuildRelationship)
        cls.register_model(Neo4jModelType.USES_MATERIAL.value, ProcessMaterialRelationship)
        cls.register_model(Neo4jModelType.HAS_QUALITY.value, ProcessQualityRelationship)
        cls.register_model(Neo4jModelType.MONITORED_BY.value, ProcessSensorRelationship)
        cls.register_model(Neo4jModelType.OPERATED_BY.value, ProcessOperatorRelationship)
        cls.register_model(Neo4jModelType.GENERATES_ALERT.value, ProcessAlertRelationship)
        cls.register_model(Neo4jModelType.HAS_DEFECT.value, ProcessDefectRelationship)
        cls.register_model(Neo4jModelType.CAPTURED_BY.value, ProcessImageRelationship)
        cls.register_model(Neo4jModelType.LOGGED_IN.value, ProcessLogRelationship)
    
    @classmethod
    def register_model(cls, model_type: str, model_class: Type[BaseModel]):
        """
        Register a model class.
        
        Args:
            model_type: Model type string
            model_class: Model class
        """
        cls._models[model_type] = model_class
    
    @classmethod
    def get_model_class(cls, model_type: str) -> Optional[Type[BaseModel]]:
        """
        Get model class by type.
        
        Args:
            model_type: Model type string
            
        Returns:
            Model class or None if not found
        """
        if not cls._models:
            cls._register_default_models()
        
        return cls._models.get(model_type)
    
    @classmethod
    def ensure_models_loaded(cls):
        """Ensure all default models are loaded."""
        if not cls._models:
            cls._register_default_models()
    
    @classmethod
    def create_model(cls, model_type: str, data: Dict[str, Any]) -> BaseModel:
        """
        Create a model instance.
        
        Args:
            model_type: Model type string
            data: Model data dictionary
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model type is not found
            ValidationError: If data validation fails
        """
        model_class = cls.get_model_class(model_type)
        if not model_class:
            raise ValueError(f"Unknown model type: {model_type}")
        
        try:
            return model_class(**data)
        except PydanticValidationError as e:
            raise PydanticValidationError(f"Validation failed for {model_type}: {e}")
    
    @classmethod
    def validate_data(cls, model_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data against a model schema.
        
        Args:
            model_type: Model type string
            data: Data dictionary to validate
            
        Returns:
            Validation result dictionary
            
        Raises:
            ValueError: If model type is not found
        """
        model_class = cls.get_model_class(model_type)
        if not model_class:
            raise ValueError(f"Unknown model type: {model_type}")
        
        try:
            model_instance = model_class(**data)
            return {
                "valid": True,
                "model_type": model_type,
                "data": model_instance.model_dump(),
                "errors": []
            }
        except PydanticValidationError as e:
            return {
                "valid": False,
                "model_type": model_type,
                "data": data,
                "errors": [str(error) for error in e.errors()]
            }
    
    @classmethod
    def get_model_schema(cls, model_type: str) -> Dict[str, Any]:
        """
        Get model schema.
        
        Args:
            model_type: Model type string
            
        Returns:
            Model schema dictionary
            
        Raises:
            ValueError: If model type is not found
        """
        model_class = cls.get_model_class(model_type)
        if not model_class:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model_class.schema()
    
    @classmethod
    def get_all_model_types(cls) -> List[str]:
        """
        Get all registered model types.
        
        Returns:
            List of model type strings
        """
        if not cls._models:
            cls._register_default_models()
        
        return list(cls._models.keys())
    
    @classmethod
    def get_node_model_types(cls) -> List[str]:
        """
        Get all node model types.
        
        Returns:
            List of node model type strings
        """
        node_types = [
            Neo4jModelType.PROCESS.value,
            Neo4jModelType.MACHINE.value,
            Neo4jModelType.PART.value,
            Neo4jModelType.BUILD.value,
            Neo4jModelType.MATERIAL.value,
            Neo4jModelType.QUALITY.value,
            Neo4jModelType.SENSOR.value,
            Neo4jModelType.USER.value,
            Neo4jModelType.OPERATOR.value,
            Neo4jModelType.ALERT.value,
            Neo4jModelType.DEFECT.value,
            Neo4jModelType.IMAGE.value,
            Neo4jModelType.LOG.value,
            Neo4jModelType.INSPECTION.value
        ]
        return node_types
    
    @classmethod
    def get_relationship_model_types(cls) -> List[str]:
        """
        Get all relationship model types.
        
        Returns:
            List of relationship model type strings
        """
        relationship_types = [
            Neo4jModelType.USES_MACHINE.value,
            Neo4jModelType.CREATES_PART.value,
            Neo4jModelType.PART_OF_BUILD.value,
            Neo4jModelType.USES_MATERIAL.value,
            Neo4jModelType.HAS_QUALITY.value,
            Neo4jModelType.MONITORED_BY.value,
            Neo4jModelType.OPERATED_BY.value,
            Neo4jModelType.GENERATES_ALERT.value,
            Neo4jModelType.HAS_DEFECT.value,
            Neo4jModelType.CAPTURED_BY.value,
            Neo4jModelType.LOGGED_IN.value
        ]
        return relationship_types
    
    @classmethod
    def validate_graph_data(cls, nodes: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate complete graph data.
        
        Args:
            nodes: List of node data dictionaries
            relationships: List of relationship data dictionaries
            
        Returns:
            Validation result dictionary
        """
        results = {
            "valid": True,
            "node_results": [],
            "relationship_results": [],
            "errors": [],
            "warnings": []
        }
        
        # Validate nodes
        for node in nodes:
            node_type = node.get('node_type', 'Unknown')
            try:
                result = cls.validate_data(node_type, node)
                results["node_results"].append(result)
                if not result["valid"]:
                    results["valid"] = False
                    results["errors"].extend(result["errors"])
            except Exception as e:
                results["valid"] = False
                results["errors"].append(f"Node validation error: {str(e)}")
        
        # Validate relationships
        for relationship in relationships:
            relationship_type = relationship.get('relationship_type', 'Unknown')
            try:
                result = cls.validate_data(relationship_type, relationship)
                results["relationship_results"].append(result)
                if not result["valid"]:
                    results["valid"] = False
                    results["errors"].extend(result["errors"])
            except Exception as e:
                results["valid"] = False
                results["errors"].append(f"Relationship validation error: {str(e)}")
        
        return results
    
    @classmethod
    def create_graph_models(cls, nodes: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create model instances for graph data.
        
        Args:
            nodes: List of node data dictionaries
            relationships: List of relationship data dictionaries
            
        Returns:
            Dictionary with created models and validation results
        """
        results = {
            "nodes": [],
            "relationships": [],
            "errors": [],
            "warnings": []
        }
        
        # Create node models
        for node in nodes:
            node_type = node.get('node_type', 'Unknown')
            try:
                model_instance = cls.create_model(node_type, node)
                results["nodes"].append(model_instance)
            except Exception as e:
                results["errors"].append(f"Node creation error for {node_type}: {str(e)}")
        
        # Create relationship models
        for relationship in relationships:
            relationship_type = relationship.get('relationship_type', 'Unknown')
            try:
                model_instance = cls.create_model(relationship_type, relationship)
                results["relationships"].append(model_instance)
            except Exception as e:
                results["errors"].append(f"Relationship creation error for {relationship_type}: {str(e)}")
        
        return results
    
    @classmethod
    def export_models_to_dict(cls, models: List[BaseModel]) -> List[Dict[str, Any]]:
        """
        Export models to dictionary format.
        
        Args:
            models: List of model instances
            
        Returns:
            List of model dictionaries
        """
        return [model.model_dump() for model in models]
    
    @classmethod
    def export_models_to_json(cls, models: List[BaseModel]) -> str:
        """
        Export models to JSON format.
        
        Args:
            models: List of model instances
            
        Returns:
            JSON string
        """
        model_dicts = cls.export_models_to_dict(models)
        return json.dumps(model_dicts, indent=2, default=str)
    
    @classmethod
    def import_models_from_dict(cls, model_dicts: List[Dict[str, Any]], model_type: str) -> List[BaseModel]:
        """
        Import models from dictionary format.
        
        Args:
            model_dicts: List of model dictionaries
            model_type: Model type string
            
        Returns:
            List of model instances
        """
        models = []
        for model_dict in model_dicts:
            try:
                model_instance = cls.create_model(model_type, model_dict)
                models.append(model_instance)
            except Exception as e:
                raise ValueError(f"Failed to import model: {str(e)}")
        
        return models
    
    @classmethod
    def import_models_from_json(cls, json_string: str, model_type: str) -> List[BaseModel]:
        """
        Import models from JSON format.
        
        Args:
            json_string: JSON string
            model_type: Model type string
            
        Returns:
            List of model instances
        """
        model_dicts = safe_json_loads_with_fallback(json_string, 'json_string', 5000, [])
        return cls.import_models_from_dict(model_dicts, model_type)
