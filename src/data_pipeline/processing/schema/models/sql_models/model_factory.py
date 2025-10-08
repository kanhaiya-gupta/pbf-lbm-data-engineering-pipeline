"""
Model Factory

This module provides a factory for creating and managing PBF-LB/M data models dynamically.
"""

from typing import Any, Dict, List, Optional, Type, Union, get_type_hints
from datetime import datetime
import importlib
import inspect
from enum import Enum
import json

from .base_model import BaseDataModel
from .pbf_process_model import PBFProcessModel
from .ispm_monitoring_model import ISPMMonitoringModel
from .ct_scan_model import CTScanModel
from .powder_bed_model import PowderBedModel

class ModelType(Enum):
    """Enumeration of available model types."""
    PBF_PROCESS = "pbf_process"
    ISPM_MONITORING = "ispm_monitoring"
    CT_SCAN = "ct_scan"
    POWDER_BED = "powder_bed"

class ModelFactory:
    """
    Factory class for creating and managing PBF-LB/M data models.
    
    This factory provides methods to create model instances, validate data,
    and manage model registrations dynamically.
    """
    
    # Registry of available models
    _model_registry: Dict[str, Type[BaseDataModel]] = {
        ModelType.PBF_PROCESS.value: PBFProcessModel,
        ModelType.ISPM_MONITORING.value: ISPMMonitoringModel,
        ModelType.CT_SCAN.value: CTScanModel,
        ModelType.POWDER_BED.value: PowderBedModel
    }
    
    # Model metadata cache
    _model_metadata: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register_model(cls, model_type: str, model_class: Type[BaseDataModel]) -> None:
        """
        Register a new model type.
        
        Args:
            model_type: String identifier for the model type
            model_class: The model class to register
        """
        if not issubclass(model_class, BaseDataModel):
            raise ValueError(f"Model class must inherit from BaseDataModel")
        
        cls._model_registry[model_type] = model_class
        cls._model_metadata[model_type] = cls._extract_model_metadata(model_class)
    
    @classmethod
    def unregister_model(cls, model_type: str) -> None:
        """
        Unregister a model type.
        
        Args:
            model_type: String identifier for the model type
        """
        if model_type in cls._model_registry:
            del cls._model_registry[model_type]
        if model_type in cls._model_metadata:
            del cls._model_metadata[model_type]
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """
        Get list of available model types.
        
        Returns:
            List of available model type strings
        """
        return list(cls._model_registry.keys())
    
    @classmethod
    def is_model_available(cls, model_type: str) -> bool:
        """
        Check if a model type is available.
        
        Args:
            model_type: String identifier for the model type
            
        Returns:
            True if model is available, False otherwise
        """
        return model_type in cls._model_registry
    
    @classmethod
    def create_model(cls, model_type: str, data: Dict[str, Any]) -> BaseDataModel:
        """
        Create a model instance from data.
        
        Args:
            model_type: String identifier for the model type
            data: Dictionary containing model data
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model type is not available
            ValidationError: If data validation fails
        """
        if not cls.is_model_available(model_type):
            raise ValueError(f"Model type '{model_type}' is not available. Available types: {cls.get_available_models()}")
        
        model_class = cls._model_registry[model_type]
        return model_class(**data)
    
    @classmethod
    def create_model_from_json(cls, model_type: str, json_data: str) -> BaseDataModel:
        """
        Create a model instance from JSON string.
        
        Args:
            model_type: String identifier for the model type
            json_data: JSON string containing model data
            
        Returns:
            Model instance
        """
        data = json.loads(json_data)
        return cls.create_model(model_type, data)
    
    @classmethod
    def create_model_from_dict(cls, model_type: str, data: Dict[str, Any]) -> BaseDataModel:
        """
        Create a model instance from dictionary.
        
        Args:
            model_type: String identifier for the model type
            data: Dictionary containing model data
            
        Returns:
            Model instance
        """
        return cls.create_model(model_type, data)
    
    @classmethod
    def create_empty_model(cls, model_type: str) -> BaseDataModel:
        """
        Create an empty model instance with default values.
        
        Args:
            model_type: String identifier for the model type
            
        Returns:
            Empty model instance
        """
        if not cls.is_model_available(model_type):
            raise ValueError(f"Model type '{model_type}' is not available")
        
        model_class = cls._model_registry[model_type]
        return model_class()
    
    @classmethod
    def validate_data(cls, model_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data against a model schema without creating an instance.
        
        Args:
            model_type: String identifier for the model type
            data: Dictionary containing data to validate
            
        Returns:
            Dictionary containing validation results
        """
        if not cls.is_model_available(model_type):
            raise ValueError(f"Model type '{model_type}' is not available")
        
        model_class = cls._model_registry[model_type]
        
        try:
            # Try to create the model to validate
            model = model_class(**data)
            return {
                'valid': True,
                'errors': [],
                'warnings': [],
                'model_data': model.dict()
            }
        except Exception as e:
            return {
                'valid': False,
                'errors': [str(e)],
                'warnings': [],
                'model_data': None
            }
    
    @classmethod
    def get_model_schema(cls, model_type: str) -> Dict[str, Any]:
        """
        Get the JSON schema for a model type.
        
        Args:
            model_type: String identifier for the model type
            
        Returns:
            JSON schema dictionary
        """
        if not cls.is_model_available(model_type):
            raise ValueError(f"Model type '{model_type}' is not available")
        
        model_class = cls._model_registry[model_type]
        return model_class.schema()
    
    @classmethod
    def get_model_fields(cls, model_type: str) -> Dict[str, Any]:
        """
        Get field information for a model type.
        
        Args:
            model_type: String identifier for the model type
            
        Returns:
            Dictionary containing field information
        """
        if not cls.is_model_available(model_type):
            raise ValueError(f"Model type '{model_type}' is not available")
        
        model_class = cls._model_registry[model_type]
        return {
            field_name: {
                'type': str(field_info.type_),
                'required': field_info.is_required(),
                'default': field_info.default,
                'description': field_info.field_info.description
            }
            for field_name, field_info in model_class.__fields__.items()
        }
    
    @classmethod
    def get_required_fields(cls, model_type: str) -> List[str]:
        """
        Get list of required fields for a model type.
        
        Args:
            model_type: String identifier for the model type
            
        Returns:
            List of required field names
        """
        fields = cls.get_model_fields(model_type)
        return [field_name for field_name, field_info in fields.items() 
                if field_info['required']]
    
    @classmethod
    def get_optional_fields(cls, model_type: str) -> List[str]:
        """
        Get list of optional fields for a model type.
        
        Args:
            model_type: String identifier for the model type
            
        Returns:
            List of optional field names
        """
        fields = cls.get_model_fields(model_type)
        return [field_name for field_name, field_info in fields.items() 
                if not field_info['required']]
    
    @classmethod
    def convert_data_types(cls, model_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert data types according to model schema.
        
        Args:
            model_type: String identifier for the model type
            data: Dictionary containing data to convert
            
        Returns:
            Dictionary with converted data types
        """
        if not cls.is_model_available(model_type):
            raise ValueError(f"Model type '{model_type}' is not available")
        
        model_class = cls._model_registry[model_type]
        converted_data = {}
        
        for field_name, field_info in model_class.__fields__.items():
            if field_name in data:
                value = data[field_name]
                field_type = field_info.type_
                
                # Convert based on field type
                if field_type == datetime:
                    if isinstance(value, str):
                        try:
                            converted_data[field_name] = datetime.fromisoformat(value.replace('Z', '+00:00'))
                        except ValueError:
                            converted_data[field_name] = value
                    else:
                        converted_data[field_name] = value
                elif field_type == bool:
                    if isinstance(value, str):
                        converted_data[field_name] = value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        converted_data[field_name] = bool(value)
                elif field_type == int:
                    try:
                        converted_data[field_name] = int(value)
                    except (ValueError, TypeError):
                        converted_data[field_name] = value
                elif field_type == float:
                    try:
                        converted_data[field_name] = float(value)
                    except (ValueError, TypeError):
                        converted_data[field_name] = value
                else:
                    converted_data[field_name] = value
            else:
                # Use default value if available
                if field_info.default is not None:
                    converted_data[field_name] = field_info.default
        
        return converted_data
    
    @classmethod
    def merge_models(cls, model1: BaseDataModel, model2: BaseDataModel) -> BaseDataModel:
        """
        Merge two models of the same type.
        
        Args:
            model1: First model instance
            model2: Second model instance
            
        Returns:
            Merged model instance
        """
        if type(model1) != type(model2):
            raise ValueError("Cannot merge models of different types")
        
        return model1.merge_with(model2)
    
    @classmethod
    def batch_create_models(cls, model_type: str, data_list: List[Dict[str, Any]]) -> List[BaseDataModel]:
        """
        Create multiple model instances from a list of data dictionaries.
        
        Args:
            model_type: String identifier for the model type
            data_list: List of dictionaries containing model data
            
        Returns:
            List of model instances
        """
        if not cls.is_model_available(model_type):
            raise ValueError(f"Model type '{model_type}' is not available")
        
        models = []
        for data in data_list:
            try:
                model = cls.create_model(model_type, data)
                models.append(model)
            except Exception as e:
                # Log error and continue with next model
                print(f"Error creating model from data {data}: {e}")
                continue
        
        return models
    
    @classmethod
    def batch_validate_models(cls, model_type: str, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate multiple data dictionaries against a model schema.
        
        Args:
            model_type: String identifier for the model type
            data_list: List of dictionaries containing data to validate
            
        Returns:
            List of validation result dictionaries
        """
        results = []
        for i, data in enumerate(data_list):
            result = cls.validate_data(model_type, data)
            result['index'] = i
            results.append(result)
        
        return results
    
    @classmethod
    def get_model_statistics(cls, model_type: str, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about a collection of data for a model type.
        
        Args:
            model_type: String identifier for the model type
            data_list: List of dictionaries containing data
            
        Returns:
            Dictionary containing statistics
        """
        if not data_list:
            return {
                'total_records': 0,
                'valid_records': 0,
                'invalid_records': 0,
                'validation_rate': 0.0,
                'field_completeness': {},
                'common_errors': []
            }
        
        validation_results = cls.batch_validate_models(model_type, data_list)
        valid_count = sum(1 for result in validation_results if result['valid'])
        invalid_count = len(validation_results) - valid_count
        
        # Calculate field completeness
        field_completeness = {}
        fields = cls.get_model_fields(model_type)
        
        for field_name in fields:
            non_null_count = sum(1 for data in data_list if data.get(field_name) is not None)
            field_completeness[field_name] = non_null_count / len(data_list)
        
        # Collect common errors
        error_counts = {}
        for result in validation_results:
            if not result['valid']:
                for error in result['errors']:
                    error_counts[error] = error_counts.get(error, 0) + 1
        
        common_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_records': len(data_list),
            'valid_records': valid_count,
            'invalid_records': invalid_count,
            'validation_rate': valid_count / len(data_list),
            'field_completeness': field_completeness,
            'common_errors': common_errors
        }
    
    @classmethod
    def _extract_model_metadata(cls, model_class: Type[BaseDataModel]) -> Dict[str, Any]:
        """
        Extract metadata from a model class.
        
        Args:
            model_class: The model class to extract metadata from
            
        Returns:
            Dictionary containing model metadata
        """
        return {
            'class_name': model_class.__name__,
            'module': model_class.__module__,
            'fields': list(model_class.__fields__.keys()),
            'required_fields': [field_name for field_name, field_info in model_class.__fields__.items() 
                              if field_info.is_required()],
            'optional_fields': [field_name for field_name, field_info in model_class.__fields__.items() 
                              if not field_info.is_required()],
            'primary_key': model_class.get_primary_key() if hasattr(model_class, 'get_primary_key') else None,
            'description': model_class.__doc__ or '',
            'created_at': datetime.utcnow().isoformat()
        }
    
    @classmethod
    def get_model_metadata(cls, model_type: str) -> Dict[str, Any]:
        """
        Get metadata for a model type.
        
        Args:
            model_type: String identifier for the model type
            
        Returns:
            Dictionary containing model metadata
        """
        if not cls.is_model_available(model_type):
            raise ValueError(f"Model type '{model_type}' is not available")
        
        if model_type not in cls._model_metadata:
            model_class = cls._model_registry[model_type]
            cls._model_metadata[model_type] = cls._extract_model_metadata(model_class)
        
        return cls._model_metadata[model_type]
    
    @classmethod
    def get_all_model_metadata(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for all registered models.
        
        Returns:
            Dictionary containing metadata for all models
        """
        metadata = {}
        for model_type in cls._model_registry.keys():
            metadata[model_type] = cls.get_model_metadata(model_type)
        
        return metadata
    
    @classmethod
    def export_model_schemas(cls, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Export all model schemas to a dictionary or file.
        
        Args:
            output_file: Optional file path to save schemas
            
        Returns:
            Dictionary containing all model schemas
        """
        schemas = {}
        for model_type in cls._model_registry.keys():
            schemas[model_type] = cls.get_model_schema(model_type)
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(schemas, f, indent=2, default=str)
        
        return schemas
    
    @classmethod
    def import_model_schemas(cls, schemas_file: str) -> None:
        """
        Import model schemas from a file.
        
        Args:
            schemas_file: Path to the schemas file
        """
        with open(schemas_file, 'r') as f:
            schemas = json.load(f)
        
        # This would require implementing schema-to-model conversion
        # For now, this is a placeholder for future functionality
        raise NotImplementedError("Schema import functionality not yet implemented")

# Convenience functions for common operations
def create_model(model_type: str, data: Dict[str, Any]) -> BaseDataModel:
    """Convenience function to create a model instance."""
    return ModelFactory.create_model(model_type, data)

def validate_data(model_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to validate data against a model schema."""
    return ModelFactory.validate_data(model_type, data)

def get_model_schema(model_type: str) -> Dict[str, Any]:
    """Convenience function to get model schema."""
    return ModelFactory.get_model_schema(model_type)

def get_available_models() -> List[str]:
    """Convenience function to get available model types."""
    return ModelFactory.get_available_models()
