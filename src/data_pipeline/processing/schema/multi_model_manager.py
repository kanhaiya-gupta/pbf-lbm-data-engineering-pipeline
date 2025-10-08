"""
Multi-Model Schema Manager

This module provides unified management of schemas across different NoSQL data models
for the PBF-LB/M data pipeline, including MongoDB, Redis, Cassandra, Elasticsearch, and Neo4j.
"""

import json
import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class DataModelType(Enum):
    """Enumeration of supported data model types."""
    DOCUMENT = "document"  # MongoDB
    KEY_VALUE = "key_value"  # Redis
    COLUMNAR = "columnar"  # Cassandra
    SEARCH = "search"  # Elasticsearch
    GRAPH = "graph"  # Neo4j
    SQL = "sql"  # PostgreSQL, etc.


class SchemaFormat(Enum):
    """Enumeration of supported schema formats."""
    JSON = "json"
    AVRO = "avro"
    SQL = "sql"
    CQL = "cql"
    CYPHER = "cypher"


class MultiModelManager:
    """
    Unified manager for multi-model schema management across different NoSQL databases.
    
    Provides a consistent interface for schema registration, validation, evolution,
    and cross-model transformations for the PBF-LB/M data pipeline.
    """
    
    def __init__(self, schema_base_path: Optional[str] = None):
        """
        Initialize the multi-model manager.
        
        Args:
            schema_base_path: Base path for schema files
        """
        if schema_base_path is None:
            schema_base_path = os.path.join(
                os.path.dirname(__file__), 
                "nosql_schemas"
            )
        
        self.schema_base_path = Path(schema_base_path)
        self.registered_schemas: Dict[str, Dict[str, Any]] = {}
        self.model_mappings: Dict[str, DataModelType] = {}
        self.cross_model_relationships: Dict[str, List[str]] = {}
        
        self._initialize_schema_directories()
        self._load_existing_schemas()
    
    def _initialize_schema_directories(self):
        """Initialize schema directory structure."""
        try:
            directories = [
                "mongodb_schemas",
                "redis_schemas", 
                "cassandra_schemas",
                "elasticsearch_schemas",
                "neo4j_schemas"
            ]
            
            for directory in directories:
                dir_path = self.schema_base_path / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                
            logger.info("Schema directories initialized")
            
        except Exception as e:
            logger.error(f"Error initializing schema directories: {e}")
    
    def _load_existing_schemas(self):
        """Load existing schemas from files."""
        try:
            for model_type in DataModelType:
                if model_type == DataModelType.SQL:
                    continue  # Skip SQL schemas for now
                    
                schema_dir = self._get_schema_directory(model_type)
                if schema_dir.exists():
                    self._load_schemas_from_directory(schema_dir, model_type)
            
            logger.info(f"Loaded {len(self.registered_schemas)} schemas")
            
        except Exception as e:
            logger.error(f"Error loading existing schemas: {e}")
    
    def _get_schema_directory(self, model_type: DataModelType) -> Path:
        """Get schema directory for a specific model type."""
        directory_mapping = {
            DataModelType.DOCUMENT: "mongodb_schemas",
            DataModelType.KEY_VALUE: "redis_schemas",
            DataModelType.COLUMNAR: "cassandra_schemas",
            DataModelType.SEARCH: "elasticsearch_schemas",
            DataModelType.GRAPH: "neo4j_schemas"
        }
        
        return self.schema_base_path / directory_mapping[model_type]
    
    def _load_schemas_from_directory(self, directory: Path, model_type: DataModelType):
        """Load schemas from a specific directory."""
        try:
            for schema_file in directory.glob("*"):
                if schema_file.is_file():
                    schema_name = schema_file.stem
                    schema_data = self._load_schema_file(schema_file)
                    
                    if schema_data:
                        self.register_schema(
                            schema_name, 
                            model_type, 
                            schema_data,
                            source_file=str(schema_file)
                        )
                        
        except Exception as e:
            logger.error(f"Error loading schemas from {directory}: {e}")
    
    def _load_schema_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load schema data from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix == '.json':
                    return json.load(f)
                elif file_path.suffix == '.cql':
                    return {"cql_schema": f.read()}
                elif file_path.suffix == '.cypher':
                    return {"cypher_schema": f.read()}
                else:
                    return {"raw_content": f.read()}
                    
        except Exception as e:
            logger.error(f"Error loading schema file {file_path}: {e}")
            return None
    
    def register_schema(self, schema_name: str, model_type: DataModelType, 
                       schema_data: Dict[str, Any], 
                       source_file: Optional[str] = None) -> bool:
        """
        Register a new schema.
        
        Args:
            schema_name: Name of the schema
            model_type: Type of data model
            schema_data: Schema definition data
            source_file: Source file path (optional)
            
        Returns:
            bool: True if registration successful
        """
        try:
            schema_id = f"{model_type.value}_{schema_name}"
            
            self.registered_schemas[schema_id] = {
                "name": schema_name,
                "model_type": model_type.value,
                "data": schema_data,
                "source_file": source_file,
                "registered_at": datetime.utcnow().isoformat(),
                "version": schema_data.get("version", "1.0.0")
            }
            
            self.model_mappings[schema_name] = model_type
            
            logger.info(f"Registered schema: {schema_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering schema {schema_name}: {e}")
            return False
    
    def get_schema(self, schema_name: str, model_type: Optional[DataModelType] = None) -> Optional[Dict[str, Any]]:
        """
        Get a registered schema.
        
        Args:
            schema_name: Name of the schema
            model_type: Type of data model (optional)
            
        Returns:
            Dict: Schema data if found
        """
        try:
            if model_type:
                schema_id = f"{model_type.value}_{schema_name}"
            else:
                # Search across all model types
                for mt in DataModelType:
                    schema_id = f"{mt.value}_{schema_name}"
                    if schema_id in self.registered_schemas:
                        return self.registered_schemas[schema_id]
                return None
            
            return self.registered_schemas.get(schema_id)
            
        except Exception as e:
            logger.error(f"Error getting schema {schema_name}: {e}")
            return None
    
    def list_schemas(self, model_type: Optional[DataModelType] = None) -> List[str]:
        """
        List all registered schemas.
        
        Args:
            model_type: Filter by model type (optional)
            
        Returns:
            List[str]: List of schema names
        """
        try:
            if model_type:
                prefix = f"{model_type.value}_"
                return [
                    schema_id[len(prefix):] 
                    for schema_id in self.registered_schemas.keys()
                    if schema_id.startswith(prefix)
                ]
            else:
                return list(self.registered_schemas.keys())
                
        except Exception as e:
            logger.error(f"Error listing schemas: {e}")
            return []
    
    def validate_data_against_schema(self, data: Dict[str, Any], 
                                   schema_name: str, 
                                   model_type: DataModelType) -> Dict[str, Any]:
        """
        Validate data against a specific schema.
        
        Args:
            data: Data to validate
            schema_name: Name of the schema
            model_type: Type of data model
            
        Returns:
            Dict: Validation results
        """
        try:
            schema = self.get_schema(schema_name, model_type)
            if not schema:
                return {
                    "valid": False,
                    "errors": [f"Schema {schema_name} not found for model type {model_type.value}"]
                }
            
            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "schema_name": schema_name,
                "model_type": model_type.value
            }
            
            # Model-specific validation
            if model_type == DataModelType.DOCUMENT:
                validation_result = self._validate_document_data(data, schema, validation_result)
            elif model_type == DataModelType.KEY_VALUE:
                validation_result = self._validate_key_value_data(data, schema, validation_result)
            elif model_type == DataModelType.COLUMNAR:
                validation_result = self._validate_columnar_data(data, schema, validation_result)
            elif model_type == DataModelType.SEARCH:
                validation_result = self._validate_search_data(data, schema, validation_result)
            elif model_type == DataModelType.GRAPH:
                validation_result = self._validate_graph_data(data, schema, validation_result)
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating data against schema {schema_name}: {e}")
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"]
            }
    
    def _validate_document_data(self, data: Dict[str, Any], schema: Dict[str, Any], 
                               result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate document data against MongoDB schema."""
        try:
            schema_def = schema["data"].get("schema", {})
            required_fields = schema_def.get("required", [])
            
            # Check required fields
            for field in required_fields:
                if field not in data:
                    result["errors"].append(f"Missing required field: {field}")
                    result["valid"] = False
            
            # Validate field types and constraints
            properties = schema_def.get("properties", {})
            for field, value in data.items():
                if field in properties:
                    field_schema = properties[field]
                    field_validation = self._validate_field(value, field_schema, field)
                    if not field_validation["valid"]:
                        result["errors"].extend(field_validation["errors"])
                        result["valid"] = False
            
            return result
            
        except Exception as e:
            result["errors"].append(f"Document validation error: {str(e)}")
            result["valid"] = False
            return result
    
    def _validate_key_value_data(self, data: Dict[str, Any], schema: Dict[str, Any], 
                                result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate key-value data against Redis schema."""
        try:
            cache_patterns = schema["data"].get("cache_patterns", {})
            
            # Validate key patterns
            for pattern_name, pattern_config in cache_patterns.items():
                pattern = pattern_config.get("pattern", "")
                if pattern and not self._matches_key_pattern(data, pattern):
                    result["warnings"].append(f"Data doesn't match pattern: {pattern}")
            
            return result
            
        except Exception as e:
            result["errors"].append(f"Key-value validation error: {str(e)}")
            result["valid"] = False
            return result
    
    def _validate_columnar_data(self, data: Dict[str, Any], schema: Dict[str, Any], 
                               result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate columnar data against Cassandra schema."""
        try:
            # Basic validation for Cassandra row structure
            if not isinstance(data, dict):
                result["errors"].append("Columnar data must be a dictionary")
                result["valid"] = False
                return result
            
            # Check for required partition key (would need to parse CQL schema)
            cql_schema = schema["data"].get("cql_schema", "")
            if "PRIMARY KEY" in cql_schema:
                # Extract primary key fields (simplified)
                result["warnings"].append("Primary key validation not implemented")
            
            return result
            
        except Exception as e:
            result["errors"].append(f"Columnar validation error: {str(e)}")
            result["valid"] = False
            return result
    
    def _validate_search_data(self, data: Dict[str, Any], schema: Dict[str, Any], 
                             result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate search data against Elasticsearch schema."""
        try:
            mappings = schema["data"].get("mappings", {})
            properties = mappings.get("properties", {})
            
            # Validate field types
            for field, value in data.items():
                if field in properties:
                    field_mapping = properties[field]
                    field_type = field_mapping.get("type", "text")
                    
                    if not self._validate_elasticsearch_type(value, field_type):
                        result["errors"].append(f"Field {field} type mismatch: expected {field_type}")
                        result["valid"] = False
            
            return result
            
        except Exception as e:
            result["errors"].append(f"Search validation error: {str(e)}")
            result["valid"] = False
            return result
    
    def _validate_graph_data(self, data: Dict[str, Any], schema: Dict[str, Any], 
                            result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate graph data against Neo4j schema."""
        try:
            # Basic validation for graph node/relationship structure
            if "node_type" in data:
                # Validate node properties
                node_type = data["node_type"]
                properties = data.get("properties", {})
                
                if not isinstance(properties, dict):
                    result["errors"].append("Node properties must be a dictionary")
                    result["valid"] = False
            
            elif "relationship_type" in data:
                # Validate relationship properties
                relationship_type = data["relationship_type"]
                properties = data.get("properties", {})
                
                if not isinstance(properties, dict):
                    result["errors"].append("Relationship properties must be a dictionary")
                    result["valid"] = False
            
            return result
            
        except Exception as e:
            result["errors"].append(f"Graph validation error: {str(e)}")
            result["valid"] = False
            return result
    
    def _validate_field(self, value: Any, field_schema: Dict[str, Any], field_name: str) -> Dict[str, Any]:
        """Validate a single field against its schema."""
        result = {"valid": True, "errors": []}
        
        try:
            field_type = field_schema.get("type")
            
            if field_type == "string":
                if not isinstance(value, str):
                    result["errors"].append(f"Field {field_name} must be a string")
                    result["valid"] = False
            elif field_type == "number":
                if not isinstance(value, (int, float)):
                    result["errors"].append(f"Field {field_name} must be a number")
                    result["valid"] = False
            elif field_type == "integer":
                if not isinstance(value, int):
                    result["errors"].append(f"Field {field_name} must be an integer")
                    result["valid"] = False
            elif field_type == "boolean":
                if not isinstance(value, bool):
                    result["errors"].append(f"Field {field_name} must be a boolean")
                    result["valid"] = False
            
            # Validate constraints
            if "minimum" in field_schema and isinstance(value, (int, float)):
                if value < field_schema["minimum"]:
                    result["errors"].append(f"Field {field_name} below minimum: {field_schema['minimum']}")
                    result["valid"] = False
            
            if "maximum" in field_schema and isinstance(value, (int, float)):
                if value > field_schema["maximum"]:
                    result["errors"].append(f"Field {field_name} above maximum: {field_schema['maximum']}")
                    result["valid"] = False
            
            if "enum" in field_schema:
                if value not in field_schema["enum"]:
                    result["errors"].append(f"Field {field_name} not in allowed values: {field_schema['enum']}")
                    result["valid"] = False
            
        except Exception as e:
            result["errors"].append(f"Field validation error for {field_name}: {str(e)}")
            result["valid"] = False
        
        return result
    
    def _validate_elasticsearch_type(self, value: Any, field_type: str) -> bool:
        """Validate value against Elasticsearch field type."""
        type_mapping = {
            "text": lambda v: isinstance(v, str),
            "keyword": lambda v: isinstance(v, str),
            "long": lambda v: isinstance(v, int),
            "integer": lambda v: isinstance(v, int),
            "short": lambda v: isinstance(v, int),
            "byte": lambda v: isinstance(v, int),
            "double": lambda v: isinstance(v, (int, float)),
            "float": lambda v: isinstance(v, (int, float)),
            "boolean": lambda v: isinstance(v, bool),
            "date": lambda v: isinstance(v, str),  # Simplified
            "object": lambda v: isinstance(v, dict),
            "nested": lambda v: isinstance(v, list)
        }
        
        validator = type_mapping.get(field_type)
        return validator(value) if validator else True
    
    def _matches_key_pattern(self, data: Dict[str, Any], pattern: str) -> bool:
        """Check if data matches a Redis key pattern."""
        # Simplified pattern matching
        # In practice, this would be more sophisticated
        return True  # Placeholder
    
    def create_cross_model_relationship(self, source_schema: str, target_schema: str, 
                                      relationship_type: str) -> bool:
        """
        Create a relationship between schemas across different models.
        
        Args:
            source_schema: Source schema name
            target_schema: Target schema name
            relationship_type: Type of relationship
            
        Returns:
            bool: True if relationship created successfully
        """
        try:
            relationship_id = f"{source_schema}->{target_schema}"
            
            if relationship_id not in self.cross_model_relationships:
                self.cross_model_relationships[relationship_id] = []
            
            self.cross_model_relationships[relationship_id].append(relationship_type)
            
            logger.info(f"Created cross-model relationship: {relationship_id} ({relationship_type})")
            return True
            
        except Exception as e:
            logger.error(f"Error creating cross-model relationship: {e}")
            return False
    
    def get_schema_statistics(self) -> Dict[str, Any]:
        """Get statistics about registered schemas."""
        try:
            stats = {
                "total_schemas": len(self.registered_schemas),
                "by_model_type": {},
                "by_version": {},
                "cross_model_relationships": len(self.cross_model_relationships)
            }
            
            # Count by model type
            for schema_id, schema_data in self.registered_schemas.items():
                model_type = schema_data["model_type"]
                stats["by_model_type"][model_type] = stats["by_model_type"].get(model_type, 0) + 1
                
                version = schema_data["version"]
                stats["by_version"][version] = stats["by_version"].get(version, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting schema statistics: {e}")
            return {}
    
    def export_schema(self, schema_name: str, model_type: DataModelType, 
                     format: SchemaFormat) -> Optional[str]:
        """
        Export a schema in a specific format.
        
        Args:
            schema_name: Name of the schema
            model_type: Type of data model
            format: Export format
            
        Returns:
            str: Exported schema content
        """
        try:
            schema = self.get_schema(schema_name, model_type)
            if not schema:
                return None
            
            if format == SchemaFormat.JSON:
                return json.dumps(schema["data"], indent=2)
            else:
                # For other formats, return raw content
                return str(schema["data"])
                
        except Exception as e:
            logger.error(f"Error exporting schema {schema_name}: {e}")
            return None
