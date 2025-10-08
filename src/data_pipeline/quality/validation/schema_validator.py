"""
Schema Validator

This module provides schema validation capabilities for the PBF-LB/M data pipeline.
"""

import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import jsonschema
from jsonschema import validate, ValidationError

from src.data_pipeline.config.pipeline_config import get_pipeline_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SchemaValidationLevel(Enum):
    """Schema validation level enumeration."""
    STRICT = "strict"
    MODERATE = "moderate"
    RELAXED = "relaxed"

@dataclass
class SchemaValidationResult:
    """Schema validation result data class."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validated_fields: int = 0
    total_fields: int = 0
    validation_level: SchemaValidationLevel = SchemaValidationLevel.STRICT
    timestamp: datetime = field(default_factory=datetime.now)

class SchemaValidator:
    """
    Schema validation service for PBF-LB/M data pipeline.
    """
    
    def __init__(self):
        self.config = get_pipeline_config()
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self.validation_results: Dict[str, SchemaValidationResult] = {}
        
        # Initialize schemas
        self._initialize_schemas()
        
    def validate_pbf_process_schema(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> SchemaValidationResult:
        """
        Validate PBF process data against schema.
        
        Args:
            data: PBF process data to validate
            
        Returns:
            SchemaValidationResult: Validation result
        """
        try:
            logger.info("Validating PBF process data schema")
            
            if isinstance(data, list):
                return self._validate_batch_data(data, "pbf_process")
            else:
                return self._validate_single_record(data, "pbf_process")
                
        except Exception as e:
            logger.error(f"Error validating PBF process schema: {e}")
            return SchemaValidationResult(
                is_valid=False,
                errors=[f"Schema validation error: {e}"],
                validation_level=SchemaValidationLevel.STRICT
            )
    
    def validate_ispm_monitoring_schema(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> SchemaValidationResult:
        """
        Validate ISPM monitoring data against schema.
        
        Args:
            data: ISPM monitoring data to validate
            
        Returns:
            SchemaValidationResult: Validation result
        """
        try:
            logger.info("Validating ISPM monitoring data schema")
            
            if isinstance(data, list):
                return self._validate_batch_data(data, "ispm_monitoring")
            else:
                return self._validate_single_record(data, "ispm_monitoring")
                
        except Exception as e:
            logger.error(f"Error validating ISPM monitoring schema: {e}")
            return SchemaValidationResult(
                is_valid=False,
                errors=[f"Schema validation error: {e}"],
                validation_level=SchemaValidationLevel.STRICT
            )
    
    def validate_ct_scan_schema(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> SchemaValidationResult:
        """
        Validate CT scan data against schema.
        
        Args:
            data: CT scan data to validate
            
        Returns:
            SchemaValidationResult: Validation result
        """
        try:
            logger.info("Validating CT scan data schema")
            
            if isinstance(data, list):
                return self._validate_batch_data(data, "ct_scan")
            else:
                return self._validate_single_record(data, "ct_scan")
                
        except Exception as e:
            logger.error(f"Error validating CT scan schema: {e}")
            return SchemaValidationResult(
                is_valid=False,
                errors=[f"Schema validation error: {e}"],
                validation_level=SchemaValidationLevel.STRICT
            )
    
    def validate_powder_bed_schema(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> SchemaValidationResult:
        """
        Validate powder bed data against schema.
        
        Args:
            data: Powder bed data to validate
            
        Returns:
            SchemaValidationResult: Validation result
        """
        try:
            logger.info("Validating powder bed data schema")
            
            if isinstance(data, list):
                return self._validate_batch_data(data, "powder_bed")
            else:
                return self._validate_single_record(data, "powder_bed")
                
        except Exception as e:
            logger.error(f"Error validating powder bed schema: {e}")
            return SchemaValidationResult(
                is_valid=False,
                errors=[f"Schema validation error: {e}"],
                validation_level=SchemaValidationLevel.STRICT
            )
    
    def _validate_single_record(self, record: Dict[str, Any], schema_name: str) -> SchemaValidationResult:
        """Validate a single record against schema."""
        try:
            if schema_name not in self.schemas:
                return SchemaValidationResult(
                    is_valid=False,
                    errors=[f"Schema {schema_name} not found"],
                    validation_level=SchemaValidationLevel.STRICT
                )
            
            schema = self.schemas[schema_name]
            errors = []
            warnings = []
            
            # Validate using JSON Schema
            try:
                validate(instance=record, schema=schema)
            except ValidationError as e:
                errors.append(f"Schema validation error: {e.message}")
            except Exception as e:
                errors.append(f"Validation error: {e}")
            
            # Additional custom validations
            custom_errors, custom_warnings = self._custom_validation(record, schema_name)
            errors.extend(custom_errors)
            warnings.extend(custom_warnings)
            
            is_valid = len(errors) == 0
            
            result = SchemaValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                validated_fields=len(record),
                total_fields=len(schema.get("properties", {})),
                validation_level=SchemaValidationLevel.STRICT
            )
            
            self.validation_results[f"{schema_name}_{datetime.now().timestamp()}"] = result
            return result
            
        except Exception as e:
            logger.error(f"Error validating single record: {e}")
            return SchemaValidationResult(
                is_valid=False,
                errors=[f"Record validation error: {e}"],
                validation_level=SchemaValidationLevel.STRICT
            )
    
    def _validate_batch_data(self, data: List[Dict[str, Any]], schema_name: str) -> SchemaValidationResult:
        """Validate batch data against schema."""
        try:
            if not data:
                return SchemaValidationResult(
                    is_valid=True,
                    errors=[],
                    warnings=["No data to validate"],
                    validated_fields=0,
                    total_fields=0,
                    validation_level=SchemaValidationLevel.STRICT
                )
            
            all_errors = []
            all_warnings = []
            total_validated_fields = 0
            total_records = len(data)
            
            for i, record in enumerate(data):
                result = self._validate_single_record(record, schema_name)
                
                if not result.is_valid:
                    all_errors.extend([f"Record {i}: {error}" for error in result.errors])
                
                all_warnings.extend([f"Record {i}: {warning}" for warning in result.warnings])
                total_validated_fields += result.validated_fields
            
            is_valid = len(all_errors) == 0
            
            result = SchemaValidationResult(
                is_valid=is_valid,
                errors=all_errors,
                warnings=all_warnings,
                validated_fields=total_validated_fields,
                total_fields=total_validated_fields,
                validation_level=SchemaValidationLevel.STRICT
            )
            
            self.validation_results[f"{schema_name}_batch_{datetime.now().timestamp()}"] = result
            return result
            
        except Exception as e:
            logger.error(f"Error validating batch data: {e}")
            return SchemaValidationResult(
                is_valid=False,
                errors=[f"Batch validation error: {e}"],
                validation_level=SchemaValidationLevel.STRICT
            )
    
    def _custom_validation(self, record: Dict[str, Any], schema_name: str) -> tuple[List[str], List[str]]:
        """Perform custom validation logic."""
        errors = []
        warnings = []
        
        try:
            if schema_name == "pbf_process":
                errors.extend(self._validate_pbf_process_custom(record))
            elif schema_name == "ispm_monitoring":
                errors.extend(self._validate_ispm_monitoring_custom(record))
            elif schema_name == "ct_scan":
                errors.extend(self._validate_ct_scan_custom(record))
            elif schema_name == "powder_bed":
                errors.extend(self._validate_powder_bed_custom(record))
                
        except Exception as e:
            errors.append(f"Custom validation error: {e}")
        
        return errors, warnings
    
    def _validate_pbf_process_custom(self, record: Dict[str, Any]) -> List[str]:
        """Custom validation for PBF process data."""
        errors = []
        
        # Validate machine ID format
        if "machine_id" in record:
            machine_id = record["machine_id"]
            if not isinstance(machine_id, str) or not machine_id.startswith("PBF-"):
                errors.append("Machine ID must be a string starting with 'PBF-'")
        
        # Validate temperature ranges
        if "chamber_temperature" in record:
            temp = record["chamber_temperature"]
            if isinstance(temp, (int, float)) and (temp < 20 or temp > 1000):
                errors.append("Chamber temperature must be between 20 and 1000 degrees")
        
        if "build_plate_temperature" in record:
            temp = record["build_plate_temperature"]
            if isinstance(temp, (int, float)) and (temp < 20 or temp > 500):
                errors.append("Build plate temperature must be between 20 and 500 degrees")
        
        # Validate pressure range
        if "chamber_pressure" in record:
            pressure = record["chamber_pressure"]
            if isinstance(pressure, (int, float)) and (pressure < 0 or pressure > 10):
                errors.append("Chamber pressure must be between 0 and 10 bar")
        
        return errors
    
    def _validate_ispm_monitoring_custom(self, record: Dict[str, Any]) -> List[str]:
        """Custom validation for ISPM monitoring data."""
        errors = []
        
        # Validate sensor ID format
        if "sensor_id" in record:
            sensor_id = record["sensor_id"]
            if not isinstance(sensor_id, str) or not sensor_id.startswith("ISPM-"):
                errors.append("Sensor ID must be a string starting with 'ISPM-'")
        
        # Validate melt pool temperature
        if "melt_pool_temperature" in record:
            temp = record["melt_pool_temperature"]
            if isinstance(temp, (int, float)) and (temp < 1000 or temp > 3000):
                errors.append("Melt pool temperature must be between 1000 and 3000 degrees")
        
        # Validate plume intensity
        if "plume_intensity" in record:
            intensity = record["plume_intensity"]
            if isinstance(intensity, (int, float)) and (intensity < 0 or intensity > 100):
                errors.append("Plume intensity must be between 0 and 100")
        
        return errors
    
    def _validate_ct_scan_custom(self, record: Dict[str, Any]) -> List[str]:
        """Custom validation for CT scan data."""
        errors = []
        
        # Validate scan ID format
        if "scan_id" in record:
            scan_id = record["scan_id"]
            if not isinstance(scan_id, str) or not scan_id.startswith("CT-"):
                errors.append("Scan ID must be a string starting with 'CT-'")
        
        # Validate porosity percentage
        if "porosity_percentage" in record:
            porosity = record["porosity_percentage"]
            if isinstance(porosity, (int, float)) and (porosity < 0 or porosity > 100):
                errors.append("Porosity percentage must be between 0 and 100")
        
        # Validate defect count
        if "num_defects" in record:
            defects = record["num_defects"]
            if isinstance(defects, (int, float)) and defects < 0:
                errors.append("Number of defects must be non-negative")
        
        return errors
    
    def _validate_powder_bed_custom(self, record: Dict[str, Any]) -> List[str]:
        """Custom validation for powder bed data."""
        errors = []
        
        # Validate image ID format
        if "image_id" in record:
            image_id = record["image_id"]
            if not isinstance(image_id, str) or not image_id.startswith("PB-"):
                errors.append("Image ID must be a string starting with 'PB-'")
        
        # Validate layer number
        if "layer_number" in record:
            layer = record["layer_number"]
            if isinstance(layer, (int, float)) and (layer < 1 or layer > 5000):
                errors.append("Layer number must be between 1 and 5000")
        
        # Validate porosity metric
        if "porosity_metric" in record:
            porosity = record["porosity_metric"]
            if isinstance(porosity, (int, float)) and (porosity < 0 or porosity > 1):
                errors.append("Porosity metric must be between 0 and 1")
        
        return errors
    
    def _initialize_schemas(self):
        """Initialize JSON schemas for different data types."""
        try:
            # PBF Process Schema
            self.schemas["pbf_process"] = {
                "type": "object",
                "properties": {
                    "machine_id": {
                        "type": "string",
                        "pattern": "^PBF-[A-Z]{2}-\\d{3}$"
                    },
                    "event_timestamp": {
                        "type": "string",
                        "format": "date-time"
                    },
                    "chamber_temperature": {
                        "type": "number",
                        "minimum": 20,
                        "maximum": 1000
                    },
                    "build_plate_temperature": {
                        "type": "number",
                        "minimum": 20,
                        "maximum": 500
                    },
                    "chamber_pressure": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10
                    },
                    "laser_power": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1000
                    },
                    "laser_speed": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10000
                    }
                },
                "required": ["machine_id", "event_timestamp", "chamber_temperature"],
                "additionalProperties": True
            }
            
            # ISPM Monitoring Schema
            self.schemas["ispm_monitoring"] = {
                "type": "object",
                "properties": {
                    "sensor_id": {
                        "type": "string",
                        "pattern": "^ISPM-\\d{3}$"
                    },
                    "event_timestamp": {
                        "type": "string",
                        "format": "date-time"
                    },
                    "melt_pool_temperature": {
                        "type": "number",
                        "minimum": 1000,
                        "maximum": 3000
                    },
                    "plume_intensity": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 100
                    },
                    "acoustic_emissions": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1000
                    }
                },
                "required": ["sensor_id", "event_timestamp", "melt_pool_temperature"],
                "additionalProperties": True
            }
            
            # CT Scan Schema
            self.schemas["ct_scan"] = {
                "type": "object",
                "properties": {
                    "scan_id": {
                        "type": "string",
                        "pattern": "^CT-\\d{6}$"
                    },
                    "part_id": {
                        "type": "string"
                    },
                    "scan_date": {
                        "type": "string",
                        "format": "date-time"
                    },
                    "porosity_percentage": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 100
                    },
                    "num_defects": {
                        "type": "integer",
                        "minimum": 0
                    },
                    "scan_volume_mm3": {
                        "type": "number",
                        "minimum": 0
                    }
                },
                "required": ["scan_id", "part_id", "scan_date", "porosity_percentage"],
                "additionalProperties": True
            }
            
            # Powder Bed Schema
            self.schemas["powder_bed"] = {
                "type": "object",
                "properties": {
                    "image_id": {
                        "type": "string",
                        "pattern": "^PB-\\d{8}$"
                    },
                    "layer_number": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5000
                    },
                    "capture_timestamp": {
                        "type": "string",
                        "format": "date-time"
                    },
                    "image_path": {
                        "type": "string"
                    },
                    "porosity_metric": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1
                    },
                    "roughness_metric": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 100
                    }
                },
                "required": ["image_id", "layer_number", "capture_timestamp", "image_path"],
                "additionalProperties": True
            }
            
            logger.info("Initialized JSON schemas for all data types")
            
        except Exception as e:
            logger.error(f"Error initializing schemas: {e}")
    
    def get_schema(self, schema_name: str) -> Optional[Dict[str, Any]]:
        """Get a schema by name."""
        return self.schemas.get(schema_name)
    
    def add_schema(self, schema_name: str, schema: Dict[str, Any]) -> bool:
        """Add a new schema."""
        try:
            self.schemas[schema_name] = schema
            logger.info(f"Added schema: {schema_name}")
            return True
        except Exception as e:
            logger.error(f"Error adding schema {schema_name}: {e}")
            return False
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of validation results."""
        try:
            total_validations = len(self.validation_results)
            successful_validations = sum(1 for result in self.validation_results.values() if result.is_valid)
            failed_validations = total_validations - successful_validations
            
            return {
                "total_validations": total_validations,
                "successful_validations": successful_validations,
                "failed_validations": failed_validations,
                "success_rate": successful_validations / total_validations if total_validations > 0 else 0,
                "available_schemas": list(self.schemas.keys()),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting validation summary: {e}")
            return {}
