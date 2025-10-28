"""
ClickHouse Model Factory

This module provides a factory for creating and managing ClickHouse Pydantic models.
"""

from typing import Dict, Any, List, Optional, Type, Union
from enum import Enum
from pydantic import BaseModel, ValidationError
import json


class ClickHouseModelType(str, Enum):
    """ClickHouse model type enumeration."""
    # Core Operational Models
    PBF_PROCESSES = "pbf_processes"
    MACHINE_STATUS = "machine_status"
    SENSOR_READINGS = "sensor_readings"
    ANALYTICS = "analytics"
    
    # MongoDB Integration Models
    PROCESS_LOGS = "process_logs"
    MACHINE_CONFIGURATIONS = "machine_configurations"
    RAW_SENSOR_DATA = "raw_sensor_data"
    MODEL_FILES = "3d_model_files"
    BUILD_INSTRUCTIONS = "build_instructions"
    CT_SCAN_IMAGES = "ct_scan_images"
    POWDER_BED_IMAGES = "powder_bed_images"
    PROCESS_IMAGES = "process_images"
    MACHINE_BUILD_FILES = "machine_build_files"
    
    # Multi-Database Models
    REDIS_CACHE_DATA = "redis_cache_data"
    JOB_QUEUE_DATA = "job_queue_data"
    USER_SESSION_DATA = "user_session_data"
    CASSANDRA_TIME_SERIES = "cassandra_time_series"
    ISPM_MONITORING = "ispm_monitoring"


class ClickHouseModelFactory:
    """Factory class for ClickHouse models."""
    
    # Model registry
    _models: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def _register_default_models(cls):
        """Register default models."""
        from .core_operational_models import (
            PBFProcessModel, MachineStatusModel, SensorReadingModel, AnalyticsModel
        )
        from .mongodb_integration_models import (
            ProcessLogModel, MachineConfigurationModel, RawSensorDataModel,
            ModelFileModel, BuildInstructionModel, CTScanImageModel,
            PowderBedImageModel, ProcessImageModel, MachineBuildFileModel
        )
        from .multi_database_models import (
            RedisCacheDataModel, JobQueueDataModel, UserSessionDataModel,
            CassandraTimeSeriesModel, ISPMMonitoringModel
        )
        
        # Register core operational models
        cls.register_model(ClickHouseModelType.PBF_PROCESSES.value, PBFProcessModel)
        cls.register_model(ClickHouseModelType.MACHINE_STATUS.value, MachineStatusModel)
        cls.register_model(ClickHouseModelType.SENSOR_READINGS.value, SensorReadingModel)
        cls.register_model(ClickHouseModelType.ANALYTICS.value, AnalyticsModel)
        
        # Register MongoDB integration models
        cls.register_model(ClickHouseModelType.PROCESS_LOGS.value, ProcessLogModel)
        cls.register_model(ClickHouseModelType.MACHINE_CONFIGURATIONS.value, MachineConfigurationModel)
        cls.register_model(ClickHouseModelType.RAW_SENSOR_DATA.value, RawSensorDataModel)
        cls.register_model(ClickHouseModelType.MODEL_FILES.value, ModelFileModel)
        cls.register_model(ClickHouseModelType.BUILD_INSTRUCTIONS.value, BuildInstructionModel)
        cls.register_model(ClickHouseModelType.CT_SCAN_IMAGES.value, CTScanImageModel)
        cls.register_model(ClickHouseModelType.POWDER_BED_IMAGES.value, PowderBedImageModel)
        cls.register_model(ClickHouseModelType.PROCESS_IMAGES.value, ProcessImageModel)
        cls.register_model(ClickHouseModelType.MACHINE_BUILD_FILES.value, MachineBuildFileModel)
        
        # Register multi-database models
        cls.register_model(ClickHouseModelType.REDIS_CACHE_DATA.value, RedisCacheDataModel)
        cls.register_model(ClickHouseModelType.JOB_QUEUE_DATA.value, JobQueueDataModel)
        cls.register_model(ClickHouseModelType.USER_SESSION_DATA.value, UserSessionDataModel)
        cls.register_model(ClickHouseModelType.CASSANDRA_TIME_SERIES.value, CassandraTimeSeriesModel)
        cls.register_model(ClickHouseModelType.ISPM_MONITORING.value, ISPMMonitoringModel)
    
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
            Model class or None
        """
        # Auto-register models if not already done
        if not cls._models:
            cls._register_default_models()
        return cls._models.get(model_type)
    
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
        except ValidationError as e:
            raise ValidationError(f"Validation failed for {model_type}: {e}")
    
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
                "data": model_instance.dict(),
                "errors": []
            }
        except ValidationError as e:
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
        
        return {
            "model_type": model_type,
            "model_class": model_class.__name__,
            "schema": model_class.model_json_schema(),
            "fields": list(model_class.model_fields.keys()),
            "required_fields": list(model_class.model_fields.keys())
        }
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """
        Get all available model types.
        
        Returns:
            List of model type strings
        """
        # Initialize models if not already done
        if not cls._models:
            cls._register_default_models()
        return list(cls._models.keys())
    
    @classmethod
    def get_model_metadata(cls, model_type: str) -> Dict[str, Any]:
        """
        Get model metadata.
        
        Args:
            model_type: Model type string
            
        Returns:
            Model metadata dictionary
            
        Raises:
            ValueError: If model type is not found
        """
        model_class = cls.get_model_class(model_type)
        if not model_class:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return {
            "model_type": model_type,
            "model_class": model_class.__name__,
            "module": model_class.__module__,
            "docstring": model_class.__doc__,
            "field_count": len(model_class.__fields__),
            "required_fields": [field for field, info in model_class.__fields__.items() 
                              if info.is_required()]
        }
    
    @classmethod
    def batch_validate(cls, model_type: str, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate multiple data records.
        
        Args:
            model_type: Model type string
            data_list: List of data dictionaries
            
        Returns:
            Batch validation result dictionary
        """
        results = []
        valid_count = 0
        invalid_count = 0
        
        for i, data in enumerate(data_list):
            result = cls.validate_data(model_type, data)
            result["index"] = i
            results.append(result)
            
            if result["valid"]:
                valid_count += 1
            else:
                invalid_count += 1
        
        return {
            "model_type": model_type,
            "total_records": len(data_list),
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "results": results
        }
    
    @classmethod
    def get_model_field_info(cls, model_type: str, field_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model field.
        
        Args:
            model_type: Model type string
            field_name: Field name
            
        Returns:
            Field information dictionary
            
        Raises:
            ValueError: If model type or field is not found
        """
        model_class = cls.get_model_class(model_type)
        if not model_class:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if field_name not in model_class.__fields__:
            raise ValueError(f"Field '{field_name}' not found in model '{model_type}'")
        
        field_info = model_class.__fields__[field_name]
        
        return {
            "model_type": model_type,
            "field_name": field_name,
            "field_type": str(field_info.type_),
            "is_required": field_info.is_required(),
            "default_value": field_info.default,
            "description": field_info.field_info.description if hasattr(field_info.field_info, 'description') else None,
            "constraints": getattr(field_info.field_info, 'constraints', {})
        }
    
    @classmethod
    def export_model_schema(cls, model_type: str, format: str = "json") -> str:
        """
        Export model schema in specified format.
        
        Args:
            model_type: Model type string
            format: Export format ("json" or "yaml")
            
        Returns:
            Exported schema string
            
        Raises:
            ValueError: If model type is not found or format is not supported
        """
        schema = cls.get_model_schema(model_type)
        
        if format.lower() == "json":
            return json.dumps(schema, indent=2)
        elif format.lower() == "yaml":
            try:
                import yaml
                return yaml.dump(schema, default_flow_style=False)
            except ImportError:
                raise ValueError("PyYAML is required for YAML export")
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def compare_models(cls, model_type1: str, model_type2: str) -> Dict[str, Any]:
        """
        Compare two models.
        
        Args:
            model_type1: First model type
            model_type2: Second model type
            
        Returns:
            Comparison result dictionary
        """
        model1_class = cls.get_model_class(model_type1)
        model2_class = cls.get_model_class(model_type2)
        
        if not model1_class:
            raise ValueError(f"Unknown model type: {model_type1}")
        if not model2_class:
            raise ValueError(f"Unknown model type: {model_type2}")
        
        fields1 = set(model1_class.model_fields.keys())
        fields2 = set(model2_class.model_fields.keys())
        
        return {
            "model1": model_type1,
            "model2": model_type2,
            "common_fields": list(fields1.intersection(fields2)),
            "unique_to_model1": list(fields1 - fields2),
            "unique_to_model2": list(fields2 - fields1),
            "field_count_model1": len(fields1),
            "field_count_model2": len(fields2),
            "identical": fields1 == fields2
        }


# Convenience functions
def create_clickhouse_model(model_type: str, data: Dict[str, Any]) -> BaseModel:
    """
    Create a ClickHouse model instance.
    
    Args:
        model_type: Model type string
        data: Model data dictionary
        
    Returns:
        Model instance
    """
    return ClickHouseModelFactory.create_model(model_type, data)


def validate_clickhouse_data(model_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate data against a ClickHouse model schema.
    
    Args:
        model_type: Model type string
        data: Data dictionary to validate
        
    Returns:
        Validation result dictionary
    """
    return ClickHouseModelFactory.validate_data(model_type, data)


def get_clickhouse_model_schema(model_type: str) -> Dict[str, Any]:
    """
    Get ClickHouse model schema.
    
    Args:
        model_type: Model type string
        
    Returns:
        Model schema dictionary
    """
    return ClickHouseModelFactory.get_model_schema(model_type)


def get_available_clickhouse_models() -> List[str]:
    """
    Get all available ClickHouse model types.
    
    Returns:
        List of model type strings
    """
    return ClickHouseModelFactory.get_available_models()
