"""
Schema Validator

This module provides schema validation capabilities for the PBF-LB/M data pipeline.
"""

import json
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from datetime import datetime
import re

from src.data_pipeline.config.pipeline_config import get_pipeline_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SchemaValidator:
    """
    Schema validator for data validation in the PBF-LB/M data pipeline.
    """
    
    def __init__(self):
        self.config = get_pipeline_config()
        self.validation_config = self._load_validation_config()
        self.validation_rules = self._load_validation_rules()
    
    def _load_validation_config(self) -> Dict[str, Any]:
        """Load validation configuration."""
        try:
            return self.config.get('schema_validation', {
                'strict_mode': True,
                'allow_extra_fields': False,
                'allow_missing_fields': False,
                'validate_data_types': True,
                'validate_constraints': True,
                'validate_formats': True,
                'error_threshold': 0.1,  # 10% error threshold
                'warning_threshold': 0.05  # 5% warning threshold
            })
        except Exception as e:
            logger.error(f"Error loading validation configuration: {e}")
            return {}
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules for different data types."""
        try:
            return self.config.get('validation_rules', {
                'pbf_process_data': {
                    'required_fields': ['process_id', 'timestamp', 'temperature', 'pressure'],
                    'field_types': {
                        'process_id': 'string',
                        'timestamp': 'datetime',
                        'temperature': 'number',
                        'pressure': 'number',
                        'laser_power': 'number',
                        'scan_speed': 'number',
                        'layer_height': 'number'
                    },
                    'constraints': {
                        'temperature': {'min': 0, 'max': 2000},
                        'pressure': {'min': 0, 'max': 1000},
                        'laser_power': {'min': 0, 'max': 1000},
                        'scan_speed': {'min': 0, 'max': 10000},
                        'layer_height': {'min': 0.01, 'max': 1.0}
                    },
                    'formats': {
                        'timestamp': 'iso_datetime',
                        'process_id': 'uuid'
                    }
                },
                'ispm_monitoring_data': {
                    'required_fields': ['monitoring_id', 'timestamp', 'sensor_id', 'value'],
                    'field_types': {
                        'monitoring_id': 'string',
                        'timestamp': 'datetime',
                        'sensor_id': 'string',
                        'sensor_type': 'string',
                        'value': 'number',
                        'unit': 'string'
                    },
                    'constraints': {
                        'value': {'min': -1000, 'max': 1000}
                    },
                    'formats': {
                        'timestamp': 'iso_datetime',
                        'monitoring_id': 'uuid'
                    }
                },
                'ct_scan_data': {
                    'required_fields': ['scan_id', 'created_at', 'file_path', 'resolution'],
                    'field_types': {
                        'scan_id': 'string',
                        'created_at': 'datetime',
                        'file_path': 'string',
                        'resolution': 'number',
                        'file_size': 'number',
                        'format': 'string'
                    },
                    'constraints': {
                        'resolution': {'min': 1, 'max': 10000},
                        'file_size': {'min': 0, 'max': 1000000000}  # 1GB
                    },
                    'formats': {
                        'created_at': 'iso_datetime',
                        'scan_id': 'uuid'
                    }
                },
                'powder_bed_data': {
                    'required_fields': ['bed_id', 'timestamp', 'layer_number', 'image_id'],
                    'field_types': {
                        'bed_id': 'string',
                        'timestamp': 'datetime',
                        'layer_number': 'integer',
                        'image_id': 'string',
                        'resolution': 'number',
                        'file_size': 'number'
                    },
                    'constraints': {
                        'layer_number': {'min': 1, 'max': 10000},
                        'resolution': {'min': 1, 'max': 10000},
                        'file_size': {'min': 0, 'max': 1000000000}  # 1GB
                    },
                    'formats': {
                        'timestamp': 'iso_datetime',
                        'bed_id': 'uuid'
                    }
                }
            })
        except Exception as e:
            logger.error(f"Error loading validation rules: {e}")
            return {}
    
    def validate_data(self, data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame], 
                     schema_name: str) -> Dict[str, Any]:
        """Validate data against schema rules."""
        try:
            if schema_name not in self.validation_rules:
                return {'valid': False, 'error': f'Unknown schema: {schema_name}'}
            
            validation_rules = self.validation_rules[schema_name]
            
            # Handle different data types
            if isinstance(data, pd.DataFrame):
                return self._validate_dataframe(data, validation_rules, schema_name)
            elif isinstance(data, list):
                return self._validate_data_list(data, validation_rules, schema_name)
            elif isinstance(data, dict):
                return self._validate_single_record(data, validation_rules, schema_name)
            else:
                return {'valid': False, 'error': 'Unsupported data type'}
            
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            return {'valid': False, 'error': str(e)}
    
    def _validate_dataframe(self, df: pd.DataFrame, rules: Dict[str, Any], 
                          schema_name: str) -> Dict[str, Any]:
        """Validate pandas DataFrame."""
        try:
            validation_results = {
                'valid': True,
                'schema_name': schema_name,
                'total_records': len(df),
                'valid_records': 0,
                'invalid_records': 0,
                'errors': [],
                'warnings': [],
                'validation_summary': {}
            }
            
            # Check required fields
            required_fields = rules.get('required_fields', [])
            missing_fields = [field for field in required_fields if field not in df.columns]
            
            if missing_fields:
                validation_results['valid'] = False
                validation_results['errors'].append(f'Missing required fields: {missing_fields}')
            
            # Validate each record
            for index, row in df.iterrows():
                record_validation = self._validate_single_record(row.to_dict(), rules, schema_name)
                
                if record_validation['valid']:
                    validation_results['valid_records'] += 1
                else:
                    validation_results['invalid_records'] += 1
                    validation_results['errors'].extend([
                        f'Record {index}: {error}' for error in record_validation.get('errors', [])
                    ])
                
                validation_results['warnings'].extend([
                    f'Record {index}: {warning}' for warning in record_validation.get('warnings', [])
                ])
            
            # Calculate validation summary
            validation_results['validation_summary'] = self._calculate_validation_summary(validation_results)
            
            # Check error threshold
            error_rate = validation_results['invalid_records'] / validation_results['total_records'] if validation_results['total_records'] > 0 else 0
            if error_rate > self.validation_config.get('error_threshold', 0.1):
                validation_results['valid'] = False
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating DataFrame: {e}")
            return {'valid': False, 'error': str(e)}
    
    def _validate_data_list(self, data_list: List[Dict[str, Any]], rules: Dict[str, Any], 
                          schema_name: str) -> Dict[str, Any]:
        """Validate list of records."""
        try:
            validation_results = {
                'valid': True,
                'schema_name': schema_name,
                'total_records': len(data_list),
                'valid_records': 0,
                'invalid_records': 0,
                'errors': [],
                'warnings': [],
                'validation_summary': {}
            }
            
            # Validate each record
            for index, record in enumerate(data_list):
                record_validation = self._validate_single_record(record, rules, schema_name)
                
                if record_validation['valid']:
                    validation_results['valid_records'] += 1
                else:
                    validation_results['invalid_records'] += 1
                    validation_results['errors'].extend([
                        f'Record {index}: {error}' for error in record_validation.get('errors', [])
                    ])
                
                validation_results['warnings'].extend([
                    f'Record {index}: {warning}' for warning in record_validation.get('warnings', [])
                ])
            
            # Calculate validation summary
            validation_results['validation_summary'] = self._calculate_validation_summary(validation_results)
            
            # Check error threshold
            error_rate = validation_results['invalid_records'] / validation_results['total_records'] if validation_results['total_records'] > 0 else 0
            if error_rate > self.validation_config.get('error_threshold', 0.1):
                validation_results['valid'] = False
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating data list: {e}")
            return {'valid': False, 'error': str(e)}
    
    def _validate_single_record(self, record: Dict[str, Any], rules: Dict[str, Any], 
                              schema_name: str) -> Dict[str, Any]:
        """Validate a single record."""
        try:
            validation_result = {
                'valid': True,
                'errors': [],
                'warnings': []
            }
            
            # Check required fields
            required_fields = rules.get('required_fields', [])
            for field in required_fields:
                if field not in record or record[field] is None:
                    validation_result['valid'] = False
                    validation_result['errors'].append(f'Required field missing: {field}')
            
            # Validate field types
            if self.validation_config.get('validate_data_types', True):
                field_types = rules.get('field_types', {})
                for field, expected_type in field_types.items():
                    if field in record and record[field] is not None:
                        type_validation = self._validate_field_type(record[field], expected_type)
                        if not type_validation['valid']:
                            validation_result['valid'] = False
                            validation_result['errors'].append(f'Invalid type for {field}: {type_validation["error"]}')
            
            # Validate constraints
            if self.validation_config.get('validate_constraints', True):
                constraints = rules.get('constraints', {})
                for field, constraint in constraints.items():
                    if field in record and record[field] is not None:
                        constraint_validation = self._validate_constraint(record[field], constraint)
                        if not constraint_validation['valid']:
                            validation_result['valid'] = False
                            validation_result['errors'].append(f'Constraint violation for {field}: {constraint_validation["error"]}')
            
            # Validate formats
            if self.validation_config.get('validate_formats', True):
                formats = rules.get('formats', {})
                for field, expected_format in formats.items():
                    if field in record and record[field] is not None:
                        format_validation = self._validate_format(record[field], expected_format)
                        if not format_validation['valid']:
                            validation_result['warnings'].append(f'Format warning for {field}: {format_validation["warning"]}')
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating single record: {e}")
            return {'valid': False, 'errors': [str(e)], 'warnings': []}
    
    def _validate_field_type(self, value: Any, expected_type: str) -> Dict[str, Any]:
        """Validate field type."""
        try:
            type_mapping = {
                'string': str,
                'integer': int,
                'number': (int, float),
                'boolean': bool,
                'datetime': (str, datetime),
                'array': list,
                'object': dict
            }
            
            expected_python_type = type_mapping.get(expected_type)
            if not expected_python_type:
                return {'valid': False, 'error': f'Unknown type: {expected_type}'}
            
            if not isinstance(value, expected_python_type):
                return {'valid': False, 'error': f'Expected {expected_type}, got {type(value).__name__}'}
            
            return {'valid': True}
            
        except Exception as e:
            logger.error(f"Error validating field type: {e}")
            return {'valid': False, 'error': str(e)}
    
    def _validate_constraint(self, value: Any, constraint: Dict[str, Any]) -> Dict[str, Any]:
        """Validate field constraints."""
        try:
            if 'min' in constraint and value < constraint['min']:
                return {'valid': False, 'error': f'Value {value} is below minimum {constraint["min"]}'}
            
            if 'max' in constraint and value > constraint['max']:
                return {'valid': False, 'error': f'Value {value} is above maximum {constraint["max"]}'}
            
            if 'pattern' in constraint:
                pattern = constraint['pattern']
                if not re.match(pattern, str(value)):
                    return {'valid': False, 'error': f'Value {value} does not match pattern {pattern}'}
            
            if 'enum' in constraint:
                enum_values = constraint['enum']
                if value not in enum_values:
                    return {'valid': False, 'error': f'Value {value} not in allowed values {enum_values}'}
            
            return {'valid': True}
            
        except Exception as e:
            logger.error(f"Error validating constraint: {e}")
            return {'valid': False, 'error': str(e)}
    
    def _validate_format(self, value: Any, expected_format: str) -> Dict[str, Any]:
        """Validate field format."""
        try:
            if expected_format == 'iso_datetime':
                if isinstance(value, str):
                    try:
                        datetime.fromisoformat(value.replace('Z', '+00:00'))
                        return {'valid': True}
                    except ValueError:
                        return {'valid': False, 'warning': f'Invalid ISO datetime format: {value}'}
                else:
                    return {'valid': False, 'warning': f'Expected string for datetime, got {type(value).__name__}'}
            
            elif expected_format == 'uuid':
                uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
                if not re.match(uuid_pattern, str(value), re.IGNORECASE):
                    return {'valid': False, 'warning': f'Invalid UUID format: {value}'}
                return {'valid': True}
            
            elif expected_format == 'email':
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                if not re.match(email_pattern, str(value)):
                    return {'valid': False, 'warning': f'Invalid email format: {value}'}
                return {'valid': True}
            
            else:
                return {'valid': True, 'warning': f'Unknown format: {expected_format}'}
            
        except Exception as e:
            logger.error(f"Error validating format: {e}")
            return {'valid': False, 'warning': str(e)}
    
    def _calculate_validation_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate validation summary statistics."""
        try:
            total_records = validation_results['total_records']
            valid_records = validation_results['valid_records']
            invalid_records = validation_results['invalid_records']
            
            summary = {
                'total_records': total_records,
                'valid_records': valid_records,
                'invalid_records': invalid_records,
                'validity_rate': (valid_records / total_records) * 100 if total_records > 0 else 0,
                'error_rate': (invalid_records / total_records) * 100 if total_records > 0 else 0,
                'total_errors': len(validation_results['errors']),
                'total_warnings': len(validation_results['warnings'])
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error calculating validation summary: {e}")
            return {}
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        try:
            stats = {
                'configuration': self.validation_config.copy(),
                'available_schemas': list(self.validation_rules.keys()),
                'schema_details': {}
            }
            
            for schema_name, rules in self.validation_rules.items():
                stats['schema_details'][schema_name] = {
                    'required_fields': rules.get('required_fields', []),
                    'field_types': list(rules.get('field_types', {}).keys()),
                    'constraints': list(rules.get('constraints', {}).keys()),
                    'formats': list(rules.get('formats', {}).keys())
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting validation statistics: {e}")
            return {}
    
    def add_validation_rule(self, schema_name: str, rule_type: str, 
                          rule_config: Dict[str, Any]) -> Dict[str, Any]:
        """Add custom validation rule."""
        try:
            if schema_name not in self.validation_rules:
                self.validation_rules[schema_name] = {
                    'required_fields': [],
                    'field_types': {},
                    'constraints': {},
                    'formats': {}
                }
            
            if rule_type == 'required_field':
                if 'field' in rule_config:
                    self.validation_rules[schema_name]['required_fields'].append(rule_config['field'])
            
            elif rule_type == 'field_type':
                if 'field' in rule_config and 'type' in rule_config:
                    self.validation_rules[schema_name]['field_types'][rule_config['field']] = rule_config['type']
            
            elif rule_type == 'constraint':
                if 'field' in rule_config and 'constraint' in rule_config:
                    self.validation_rules[schema_name]['constraints'][rule_config['field']] = rule_config['constraint']
            
            elif rule_type == 'format':
                if 'field' in rule_config and 'format' in rule_config:
                    self.validation_rules[schema_name]['formats'][rule_config['field']] = rule_config['format']
            
            else:
                return {'status': 'error', 'error': f'Unknown rule type: {rule_type}'}
            
            return {'status': 'success', 'message': f'Validation rule added for {schema_name}'}
            
        except Exception as e:
            logger.error(f"Error adding validation rule: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def remove_validation_rule(self, schema_name: str, rule_type: str, 
                             field: str) -> Dict[str, Any]:
        """Remove validation rule."""
        try:
            if schema_name not in self.validation_rules:
                return {'status': 'error', 'error': f'Schema not found: {schema_name}'}
            
            if rule_type == 'required_field':
                if field in self.validation_rules[schema_name]['required_fields']:
                    self.validation_rules[schema_name]['required_fields'].remove(field)
            
            elif rule_type == 'field_type':
                if field in self.validation_rules[schema_name]['field_types']:
                    del self.validation_rules[schema_name]['field_types'][field]
            
            elif rule_type == 'constraint':
                if field in self.validation_rules[schema_name]['constraints']:
                    del self.validation_rules[schema_name]['constraints'][field]
            
            elif rule_type == 'format':
                if field in self.validation_rules[schema_name]['formats']:
                    del self.validation_rules[schema_name]['formats'][field]
            
            else:
                return {'status': 'error', 'error': f'Unknown rule type: {rule_type}'}
            
            return {'status': 'success', 'message': f'Validation rule removed for {schema_name}'}
            
        except Exception as e:
            logger.error(f"Error removing validation rule: {e}")
            return {'status': 'error', 'error': str(e)}
