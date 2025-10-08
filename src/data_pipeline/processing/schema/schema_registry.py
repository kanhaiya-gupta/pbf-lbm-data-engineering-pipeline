"""
Schema Registry

This module provides schema registry and versioning capabilities for the PBF-LB/M data pipeline,
including enhanced multi-model support for NoSQL databases.
"""

import json
import os
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from datetime import datetime, timedelta
import hashlib
import yaml

from src.data_pipeline.config.pipeline_config import get_pipeline_config
from .multi_model_manager import MultiModelManager, DataModelType, SchemaFormat

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SchemaRegistry:
    """
    Schema registry for managing data schemas and versions.
    """
    
    def __init__(self):
        self.config = get_pipeline_config()
        self.registry_config = self._load_registry_config()
        self.schemas = {}
        self.schema_versions = {}
        self.registry_file = self._get_registry_file_path()
        self.multi_model_manager = MultiModelManager()
        self._load_schemas()
    
    def _load_registry_config(self) -> Dict[str, Any]:
        """Load schema registry configuration."""
        try:
            return self.config.get('schema_registry', {
                'base_path': './schemas',
                'supported_formats': ['json', 'avro', 'sql'],
                'versioning_enabled': True,
                'max_versions_per_schema': 10,
                'compatibility_check_enabled': True,
                'auto_register_enabled': True,
                'schemas': {
                    'pbf_process_data': {
                        'format': 'json',
                        'version': '1.0.0',
                        'enabled': True
                    },
                    'ispm_monitoring_data': {
                        'format': 'json',
                        'version': '1.0.0',
                        'enabled': True
                    },
                    'ct_scan_data': {
                        'format': 'json',
                        'version': '1.0.0',
                        'enabled': True
                    },
                    'powder_bed_data': {
                        'format': 'json',
                        'version': '1.0.0',
                        'enabled': True
                    }
                }
            })
        except Exception as e:
            logger.error(f"Error loading schema registry configuration: {e}")
            return {}
    
    def _get_registry_file_path(self) -> str:
        """Get schema registry file path."""
        try:
            base_path = self.registry_config.get('base_path', './schemas')
            os.makedirs(base_path, exist_ok=True)
            return os.path.join(base_path, 'schema_registry.json')
        except Exception as e:
            logger.error(f"Error getting registry file path: {e}")
            return './schema_registry.json'
    
    def _load_schemas(self) -> None:
        """Load schemas from persistent storage."""
        try:
            if os.path.exists(self.registry_file):
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                    self.schemas = data.get('schemas', {})
                    self.schema_versions = data.get('versions', {})
                logger.info(f"Loaded {len(self.schemas)} schemas from registry")
            else:
                logger.info("No existing schema registry found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading schemas: {e}")
            self.schemas = {}
            self.schema_versions = {}
    
    def _save_schemas(self) -> None:
        """Save schemas to persistent storage."""
        try:
            data = {
                'schemas': self.schemas,
                'versions': self.schema_versions,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.debug("Schemas saved to registry")
        except Exception as e:
            logger.error(f"Error saving schemas: {e}")
    
    def register_schema(self, schema_name: str, schema_content: Dict[str, Any], 
                       version: Optional[str] = None, format_type: str = 'json') -> Dict[str, Any]:
        """Register a new schema or update existing one."""
        try:
            if not self.registry_config.get('auto_register_enabled', True):
                logger.warning("Schema auto-registration is disabled")
                return {'status': 'disabled', 'schema_id': None}
            
            # Generate version if not provided
            if not version:
                version = self._generate_version(schema_name)
            
            # Validate schema content
            if not self._validate_schema_content(schema_content, format_type):
                return {'status': 'error', 'error': 'Invalid schema content', 'schema_id': None}
            
            # Check compatibility if enabled
            if self.registry_config.get('compatibility_check_enabled', True):
                compatibility_result = self._check_compatibility(schema_name, schema_content, version)
                if not compatibility_result['compatible']:
                    return {
                        'status': 'error', 
                        'error': f"Schema incompatible: {compatibility_result['reason']}", 
                        'schema_id': None
                    }
            
            # Generate schema ID
            schema_id = self._generate_schema_id(schema_name, version)
            
            # Create schema record
            schema_record = {
                'schema_id': schema_id,
                'name': schema_name,
                'version': version,
                'format': format_type,
                'content': schema_content,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'hash': self._calculate_schema_hash(schema_content),
                'status': 'active'
            }
            
            # Store schema
            self.schemas[schema_id] = schema_record
            
            # Update version tracking
            if schema_name not in self.schema_versions:
                self.schema_versions[schema_name] = []
            
            self.schema_versions[schema_name].append({
                'version': version,
                'schema_id': schema_id,
                'created_at': schema_record['created_at']
            })
            
            # Sort versions
            self.schema_versions[schema_name].sort(key=lambda x: x['version'], reverse=True)
            
            # Limit versions if configured
            max_versions = self.registry_config.get('max_versions_per_schema', 10)
            if len(self.schema_versions[schema_name]) > max_versions:
                self.schema_versions[schema_name] = self.schema_versions[schema_name][:max_versions]
            
            # Save to persistent storage
            self._save_schemas()
            
            result = {
                'status': 'success',
                'schema_id': schema_id,
                'version': version,
                'message': f"Schema registered successfully: {schema_name} v{version}"
            }
            
            logger.info(f"Schema registered: {schema_name} v{version} (ID: {schema_id})")
            return result
            
        except Exception as e:
            logger.error(f"Error registering schema: {e}")
            return {'status': 'error', 'error': str(e), 'schema_id': None}
    
    def get_schema(self, schema_name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get schema by name and version."""
        try:
            if schema_name not in self.schema_versions:
                logger.warning(f"Schema not found: {schema_name}")
                return None
            
            # Get latest version if not specified
            if not version:
                version = self.schema_versions[schema_name][0]['version']
            
            # Find schema ID for the version
            schema_id = None
            for version_info in self.schema_versions[schema_name]:
                if version_info['version'] == version:
                    schema_id = version_info['schema_id']
                    break
            
            if not schema_id:
                logger.warning(f"Version {version} not found for schema {schema_name}")
                return None
            
            # Get schema record
            schema_record = self.schemas.get(schema_id)
            if not schema_record:
                logger.warning(f"Schema record not found: {schema_id}")
                return None
            
            return schema_record
            
        except Exception as e:
            logger.error(f"Error getting schema: {e}")
            return None
    
    def get_latest_schema(self, schema_name: str) -> Optional[Dict[str, Any]]:
        """Get latest version of a schema."""
        try:
            return self.get_schema(schema_name)
        except Exception as e:
            logger.error(f"Error getting latest schema: {e}")
            return None
    
    def list_schemas(self) -> List[Dict[str, Any]]:
        """List all registered schemas."""
        try:
            schema_list = []
            
            for schema_name, versions in self.schema_versions.items():
                latest_version = versions[0] if versions else None
                if latest_version:
                    schema_record = self.schemas.get(latest_version['schema_id'])
                    if schema_record:
                        schema_list.append({
                            'name': schema_name,
                            'version': latest_version['version'],
                            'format': schema_record['format'],
                            'created_at': schema_record['created_at'],
                            'updated_at': schema_record['updated_at'],
                            'status': schema_record['status']
                        })
            
            return schema_list
            
        except Exception as e:
            logger.error(f"Error listing schemas: {e}")
            return []
    
    def list_schema_versions(self, schema_name: str) -> List[Dict[str, Any]]:
        """List all versions of a schema."""
        try:
            if schema_name not in self.schema_versions:
                return []
            
            return self.schema_versions[schema_name].copy()
            
        except Exception as e:
            logger.error(f"Error listing schema versions: {e}")
            return []
    
    def validate_data(self, schema_name: str, data: Dict[str, Any], 
                     version: Optional[str] = None) -> Dict[str, Any]:
        """Validate data against a schema."""
        try:
            schema_record = self.get_schema(schema_name, version)
            if not schema_record:
                return {'valid': False, 'error': f'Schema not found: {schema_name}'}
            
            # Perform validation based on format
            format_type = schema_record['format']
            if format_type == 'json':
                return self._validate_json_data(schema_record['content'], data)
            elif format_type == 'avro':
                return self._validate_avro_data(schema_record['content'], data)
            elif format_type == 'sql':
                return self._validate_sql_data(schema_record['content'], data)
            else:
                return {'valid': False, 'error': f'Unsupported format: {format_type}'}
            
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            return {'valid': False, 'error': str(e)}
    
    def _validate_schema_content(self, schema_content: Dict[str, Any], format_type: str) -> bool:
        """Validate schema content."""
        try:
            if format_type == 'json':
                return self._validate_json_schema(schema_content)
            elif format_type == 'avro':
                return self._validate_avro_schema(schema_content)
            elif format_type == 'sql':
                return self._validate_sql_schema(schema_content)
            else:
                return False
        except Exception as e:
            logger.error(f"Error validating schema content: {e}")
            return False
    
    def _validate_json_schema(self, schema_content: Dict[str, Any]) -> bool:
        """Validate JSON schema content."""
        try:
            # Basic JSON schema validation
            required_fields = ['type', 'properties']
            return all(field in schema_content for field in required_fields)
        except Exception as e:
            logger.error(f"Error validating JSON schema: {e}")
            return False
    
    def _validate_avro_schema(self, schema_content: Dict[str, Any]) -> bool:
        """Validate Avro schema content."""
        try:
            # Basic Avro schema validation
            required_fields = ['type', 'fields']
            return all(field in schema_content for field in required_fields)
        except Exception as e:
            logger.error(f"Error validating Avro schema: {e}")
            return False
    
    def _validate_sql_schema(self, schema_content: Dict[str, Any]) -> bool:
        """Validate SQL schema content."""
        try:
            # Basic SQL schema validation
            required_fields = ['table_name', 'columns']
            return all(field in schema_content for field in required_fields)
        except Exception as e:
            logger.error(f"Error validating SQL schema: {e}")
            return False
    
    def _validate_json_data(self, schema: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against JSON schema."""
        try:
            # Simple validation - in production, use jsonschema library
            if 'properties' in schema:
                for field, field_schema in schema['properties'].items():
                    if field_schema.get('required', False) and field not in data:
                        return {'valid': False, 'error': f'Required field missing: {field}'}
                    
                    if field in data:
                        field_type = field_schema.get('type')
                        if field_type and not isinstance(data[field], self._get_python_type(field_type)):
                            return {'valid': False, 'error': f'Invalid type for field {field}: expected {field_type}'}
            
            return {'valid': True, 'message': 'Data is valid'}
            
        except Exception as e:
            logger.error(f"Error validating JSON data: {e}")
            return {'valid': False, 'error': str(e)}
    
    def _validate_avro_data(self, schema: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against Avro schema."""
        try:
            # Simple validation - in production, use avro library
            if 'fields' in schema:
                for field in schema['fields']:
                    field_name = field.get('name')
                    if field_name and field.get('required', False) and field_name not in data:
                        return {'valid': False, 'error': f'Required field missing: {field_name}'}
            
            return {'valid': True, 'message': 'Data is valid'}
            
        except Exception as e:
            logger.error(f"Error validating Avro data: {e}")
            return {'valid': False, 'error': str(e)}
    
    def _validate_sql_data(self, schema: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against SQL schema."""
        try:
            # Simple validation
            if 'columns' in schema:
                for column in schema['columns']:
                    column_name = column.get('name')
                    if column_name and column.get('not_null', False) and column_name not in data:
                        return {'valid': False, 'error': f'Required column missing: {column_name}'}
            
            return {'valid': True, 'message': 'Data is valid'}
            
        except Exception as e:
            logger.error(f"Error validating SQL data: {e}")
            return {'valid': False, 'error': str(e)}
    
    def _get_python_type(self, json_type: str):
        """Get Python type from JSON schema type."""
        type_mapping = {
            'string': str,
            'integer': int,
            'number': float,
            'boolean': bool,
            'array': list,
            'object': dict
        }
        return type_mapping.get(json_type, str)
    
    def _check_compatibility(self, schema_name: str, new_schema: Dict[str, Any], 
                           new_version: str) -> Dict[str, Any]:
        """Check schema compatibility."""
        try:
            existing_schema = self.get_latest_schema(schema_name)
            if not existing_schema:
                return {'compatible': True, 'reason': 'No existing schema'}
            
            # Simple compatibility check - in production, use proper schema evolution rules
            existing_content = existing_schema['content']
            
            # Check if required fields are still present
            if 'properties' in existing_content and 'properties' in new_schema:
                existing_props = existing_content['properties']
                new_props = new_schema['properties']
                
                for field, field_schema in existing_props.items():
                    if field_schema.get('required', False) and field not in new_props:
                        return {'compatible': False, 'reason': f'Required field removed: {field}'}
            
            return {'compatible': True, 'reason': 'Schema is compatible'}
            
        except Exception as e:
            logger.error(f"Error checking compatibility: {e}")
            return {'compatible': False, 'reason': str(e)}
    
    def _generate_version(self, schema_name: str) -> str:
        """Generate new version for schema."""
        try:
            if schema_name in self.schema_versions:
                latest_version = self.schema_versions[schema_name][0]['version']
                # Simple version increment - in production, use semantic versioning
                version_parts = latest_version.split('.')
                if len(version_parts) >= 3:
                    major, minor, patch = version_parts[0], version_parts[1], version_parts[2]
                    patch = str(int(patch) + 1)
                    return f"{major}.{minor}.{patch}"
            
            return "1.0.0"
            
        except Exception as e:
            logger.error(f"Error generating version: {e}")
            return "1.0.0"
    
    def _generate_schema_id(self, schema_name: str, version: str) -> str:
        """Generate unique schema ID."""
        try:
            return f"{schema_name}_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        except Exception as e:
            logger.error(f"Error generating schema ID: {e}")
            return f"{schema_name}_{version}_{datetime.now().isoformat()}"
    
    def _calculate_schema_hash(self, schema_content: Dict[str, Any]) -> str:
        """Calculate hash of schema content."""
        try:
            content_str = json.dumps(schema_content, sort_keys=True)
            return hashlib.md5(content_str.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating schema hash: {e}")
            return ""
    
    def get_registry_statistics(self) -> Dict[str, Any]:
        """Get schema registry statistics."""
        try:
            stats = {
                'total_schemas': len(self.schemas),
                'total_schema_names': len(self.schema_versions),
                'configuration': self.registry_config.copy(),
                'schema_summary': {}
            }
            
            for schema_name, versions in self.schema_versions.items():
                stats['schema_summary'][schema_name] = {
                    'version_count': len(versions),
                    'latest_version': versions[0]['version'] if versions else None,
                    'latest_created_at': versions[0]['created_at'] if versions else None
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting registry statistics: {e}")
            return {}
    
    def delete_schema(self, schema_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Delete a schema or specific version."""
        try:
            if schema_name not in self.schema_versions:
                return {'status': 'error', 'error': f'Schema not found: {schema_name}'}
            
            if version:
                # Delete specific version
                schema_id = None
                for i, version_info in enumerate(self.schema_versions[schema_name]):
                    if version_info['version'] == version:
                        schema_id = version_info['schema_id']
                        del self.schema_versions[schema_name][i]
                        break
                
                if schema_id and schema_id in self.schemas:
                    del self.schemas[schema_id]
                    self._save_schemas()
                    return {'status': 'success', 'message': f'Version {version} deleted for {schema_name}'}
                else:
                    return {'status': 'error', 'error': f'Version {version} not found for {schema_name}'}
            else:
                # Delete all versions
                for version_info in self.schema_versions[schema_name]:
                    schema_id = version_info['schema_id']
                    if schema_id in self.schemas:
                        del self.schemas[schema_id]
                
                del self.schema_versions[schema_name]
                self._save_schemas()
                return {'status': 'success', 'message': f'All versions deleted for {schema_name}'}
            
        except Exception as e:
            logger.error(f"Error deleting schema: {e}")
            return {'status': 'error', 'error': str(e)}
    
    # =============================================================================
    # Multi-Model NoSQL Schema Management
    # =============================================================================
    
    def register_nosql_schema(self, schema_name: str, model_type: DataModelType, 
                            schema_data: Dict[str, Any], version: str = "1.0.0") -> Dict[str, Any]:
        """
        Register a NoSQL schema with the multi-model manager.
        
        Args:
            schema_name: Name of the schema
            model_type: Type of NoSQL data model
            schema_data: Schema definition data
            version: Schema version
            
        Returns:
            Dict: Registration result
        """
        try:
            # Add version to schema data
            schema_data["version"] = version
            schema_data["registered_at"] = datetime.utcnow().isoformat()
            
            # Register with multi-model manager
            success = self.multi_model_manager.register_schema(
                schema_name, model_type, schema_data
            )
            
            if success:
                # Also register with traditional registry for compatibility
                schema_id = f"{model_type.value}_{schema_name}_{version}"
                self.schemas[schema_id] = {
                    "name": schema_name,
                    "model_type": model_type.value,
                    "version": version,
                    "data": schema_data,
                    "registered_at": datetime.utcnow().isoformat()
                }
                
                self._save_schemas()
                
                return {
                    "status": "success",
                    "message": f"NoSQL schema {schema_name} registered for {model_type.value}",
                    "schema_id": schema_id
                }
            else:
                return {
                    "status": "error",
                    "error": f"Failed to register NoSQL schema {schema_name}"
                }
                
        except Exception as e:
            logger.error(f"Error registering NoSQL schema: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_nosql_schema(self, schema_name: str, model_type: DataModelType, 
                        version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get a NoSQL schema from the multi-model manager.
        
        Args:
            schema_name: Name of the schema
            model_type: Type of NoSQL data model
            version: Schema version (optional)
            
        Returns:
            Dict: Schema data if found
        """
        try:
            return self.multi_model_manager.get_schema(schema_name, model_type)
            
        except Exception as e:
            logger.error(f"Error getting NoSQL schema: {e}")
            return None
    
    def validate_nosql_data(self, data: Dict[str, Any], schema_name: str, 
                          model_type: DataModelType) -> Dict[str, Any]:
        """
        Validate data against a NoSQL schema.
        
        Args:
            data: Data to validate
            schema_name: Name of the schema
            model_type: Type of NoSQL data model
            
        Returns:
            Dict: Validation results
        """
        try:
            return self.multi_model_manager.validate_data_against_schema(
                data, schema_name, model_type
            )
            
        except Exception as e:
            logger.error(f"Error validating NoSQL data: {e}")
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"]
            }
    
    def list_nosql_schemas(self, model_type: Optional[DataModelType] = None) -> List[str]:
        """
        List all registered NoSQL schemas.
        
        Args:
            model_type: Filter by model type (optional)
            
        Returns:
            List[str]: List of schema names
        """
        try:
            return self.multi_model_manager.list_schemas(model_type)
            
        except Exception as e:
            logger.error(f"Error listing NoSQL schemas: {e}")
            return []
    
    def create_cross_model_relationship(self, source_schema: str, target_schema: str, 
                                      relationship_type: str) -> Dict[str, Any]:
        """
        Create a relationship between schemas across different models.
        
        Args:
            source_schema: Source schema name
            target_schema: Target schema name
            relationship_type: Type of relationship
            
        Returns:
            Dict: Creation result
        """
        try:
            success = self.multi_model_manager.create_cross_model_relationship(
                source_schema, target_schema, relationship_type
            )
            
            if success:
                return {
                    "status": "success",
                    "message": f"Cross-model relationship created: {source_schema} -> {target_schema}"
                }
            else:
                return {
                    "status": "error",
                    "error": "Failed to create cross-model relationship"
                }
                
        except Exception as e:
            logger.error(f"Error creating cross-model relationship: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_multi_model_statistics(self) -> Dict[str, Any]:
        """Get statistics about multi-model schemas."""
        try:
            return self.multi_model_manager.get_schema_statistics()
            
        except Exception as e:
            logger.error(f"Error getting multi-model statistics: {e}")
            return {}
    
    def export_nosql_schema(self, schema_name: str, model_type: DataModelType, 
                           format: SchemaFormat) -> Optional[str]:
        """
        Export a NoSQL schema in a specific format.
        
        Args:
            schema_name: Name of the schema
            model_type: Type of NoSQL data model
            format: Export format
            
        Returns:
            str: Exported schema content
        """
        try:
            return self.multi_model_manager.export_schema(schema_name, model_type, format)
            
        except Exception as e:
            logger.error(f"Error exporting NoSQL schema: {e}")
            return None
    
    def get_schema_compatibility_matrix(self) -> Dict[str, Any]:
        """
        Get compatibility matrix between different schema types.
        
        Returns:
            Dict: Compatibility matrix
        """
        try:
            compatibility_matrix = {
                "document_to_key_value": {
                    "compatible": True,
                    "transformation_required": True,
                    "data_loss": "minimal"
                },
                "document_to_columnar": {
                    "compatible": True,
                    "transformation_required": True,
                    "data_loss": "moderate"
                },
                "document_to_search": {
                    "compatible": True,
                    "transformation_required": True,
                    "data_loss": "minimal"
                },
                "document_to_graph": {
                    "compatible": True,
                    "transformation_required": True,
                    "data_loss": "moderate"
                },
                "key_value_to_columnar": {
                    "compatible": False,
                    "transformation_required": True,
                    "data_loss": "significant"
                },
                "columnar_to_search": {
                    "compatible": True,
                    "transformation_required": True,
                    "data_loss": "minimal"
                },
                "graph_to_document": {
                    "compatible": True,
                    "transformation_required": True,
                    "data_loss": "moderate"
                }
            }
            
            return compatibility_matrix
            
        except Exception as e:
            logger.error(f"Error getting compatibility matrix: {e}")
            return {}
