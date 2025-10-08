"""
Schema Evolver

This module provides schema evolution capabilities for the PBF-LB/M data pipeline.
"""

import json
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from datetime import datetime, timedelta
import copy

from src.data_pipeline.config.pipeline_config import get_pipeline_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SchemaEvolver:
    """
    Schema evolver for handling schema changes and evolution.
    """
    
    def __init__(self):
        self.config = get_pipeline_config()
        self.evolution_config = self._load_evolution_config()
        self.evolution_rules = self._load_evolution_rules()
        self.evolution_history = {}
    
    def _load_evolution_config(self) -> Dict[str, Any]:
        """Load schema evolution configuration."""
        try:
            return self.config.get('schema_evolution', {
                'enabled': True,
                'auto_evolve': False,
                'backward_compatibility_required': True,
                'forward_compatibility_required': False,
                'migration_strategy': 'gradual',  # 'gradual', 'immediate', 'scheduled'
                'rollback_enabled': True,
                'evolution_window_hours': 24,
                'max_evolution_depth': 5,
                'schemas': {
                    'pbf_process_data': {
                        'enabled': True,
                        'evolution_strategy': 'gradual',
                        'compatibility_mode': 'backward'
                    },
                    'ispm_monitoring_data': {
                        'enabled': True,
                        'evolution_strategy': 'immediate',
                        'compatibility_mode': 'backward'
                    },
                    'ct_scan_data': {
                        'enabled': True,
                        'evolution_strategy': 'scheduled',
                        'compatibility_mode': 'backward'
                    },
                    'powder_bed_data': {
                        'enabled': True,
                        'evolution_strategy': 'gradual',
                        'compatibility_mode': 'backward'
                    }
                }
            })
        except Exception as e:
            logger.error(f"Error loading schema evolution configuration: {e}")
            return {}
    
    def _load_evolution_rules(self) -> Dict[str, Any]:
        """Load schema evolution rules."""
        try:
            return self.config.get('evolution_rules', {
                'field_additions': {
                    'allowed': True,
                    'default_value_required': True,
                    'nullable_allowed': True
                },
                'field_removals': {
                    'allowed': False,
                    'deprecation_period_days': 30,
                    'warning_threshold': 0.1
                },
                'field_type_changes': {
                    'allowed': True,
                    'compatible_types': {
                        'string': ['string', 'text'],
                        'integer': ['integer', 'long', 'number'],
                        'number': ['number', 'float', 'double'],
                        'boolean': ['boolean', 'string']
                    },
                    'conversion_functions': {
                        'string_to_integer': 'int',
                        'string_to_number': 'float',
                        'integer_to_string': 'str',
                        'number_to_string': 'str'
                    }
                },
                'field_renames': {
                    'allowed': True,
                    'mapping_required': True,
                    'deprecation_period_days': 30
                },
                'constraint_changes': {
                    'allowed': True,
                    'relaxation_allowed': True,
                    'tightening_allowed': False
                }
            })
        except Exception as e:
            logger.error(f"Error loading evolution rules: {e}")
            return {}
    
    def evolve_schema(self, schema_name: str, current_schema: Dict[str, Any], 
                     new_schema: Dict[str, Any], evolution_type: str = 'automatic') -> Dict[str, Any]:
        """Evolve schema from current to new version."""
        try:
            if not self._is_evolution_enabled(schema_name):
                return {'status': 'disabled', 'message': f'Schema evolution disabled for {schema_name}'}
            
            # Analyze schema changes
            changes = self._analyze_schema_changes(current_schema, new_schema)
            
            # Check compatibility
            compatibility_result = self._check_evolution_compatibility(changes, schema_name)
            
            if not compatibility_result['compatible']:
                return {
                    'status': 'error',
                    'error': f'Schema evolution not compatible: {compatibility_result["reason"]}',
                    'changes': changes
                }
            
            # Generate evolution plan
            evolution_plan = self._generate_evolution_plan(changes, schema_name, evolution_type)
            
            # Execute evolution
            evolution_result = self._execute_evolution(schema_name, current_schema, new_schema, evolution_plan)
            
            # Update evolution history
            self._update_evolution_history(schema_name, changes, evolution_result)
            
            result = {
                'status': 'success',
                'schema_name': schema_name,
                'changes': changes,
                'evolution_plan': evolution_plan,
                'evolution_result': evolution_result,
                'evolved_at': datetime.now().isoformat()
            }
            
            logger.info(f"Schema evolution completed for {schema_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error evolving schema: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _analyze_schema_changes(self, current_schema: Dict[str, Any], 
                              new_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze changes between current and new schema."""
        try:
            changes = {
                'field_additions': [],
                'field_removals': [],
                'field_type_changes': [],
                'field_renames': [],
                'constraint_changes': [],
                'format_changes': []
            }
            
            current_props = current_schema.get('properties', {})
            new_props = new_schema.get('properties', {})
            
            # Find field additions
            for field in new_props:
                if field not in current_props:
                    changes['field_additions'].append({
                        'field': field,
                        'type': new_props[field].get('type'),
                        'constraints': new_props[field].get('constraints', {}),
                        'format': new_props[field].get('format')
                    })
            
            # Find field removals
            for field in current_props:
                if field not in new_props:
                    changes['field_removals'].append({
                        'field': field,
                        'type': current_props[field].get('type'),
                        'constraints': current_props[field].get('constraints', {}),
                        'format': current_props[field].get('format')
                    })
            
            # Find field type changes
            for field in current_props:
                if field in new_props:
                    current_type = current_props[field].get('type')
                    new_type = new_props[field].get('type')
                    
                    if current_type != new_type:
                        changes['field_type_changes'].append({
                            'field': field,
                            'old_type': current_type,
                            'new_type': new_type
                        })
            
            # Find constraint changes
            for field in current_props:
                if field in new_props:
                    current_constraints = current_props[field].get('constraints', {})
                    new_constraints = new_props[field].get('constraints', {})
                    
                    if current_constraints != new_constraints:
                        changes['constraint_changes'].append({
                            'field': field,
                            'old_constraints': current_constraints,
                            'new_constraints': new_constraints
                        })
            
            # Find format changes
            for field in current_props:
                if field in new_props:
                    current_format = current_props[field].get('format')
                    new_format = new_props[field].get('format')
                    
                    if current_format != new_format:
                        changes['format_changes'].append({
                            'field': field,
                            'old_format': current_format,
                            'new_format': new_format
                        })
            
            return changes
            
        except Exception as e:
            logger.error(f"Error analyzing schema changes: {e}")
            return {}
    
    def _check_evolution_compatibility(self, changes: Dict[str, Any], schema_name: str) -> Dict[str, Any]:
        """Check if schema evolution is compatible."""
        try:
            schema_config = self.evolution_config.get('schemas', {}).get(schema_name, {})
            compatibility_mode = schema_config.get('compatibility_mode', 'backward')
            
            # Check field removals
            if changes.get('field_removals') and not self.evolution_rules['field_removals']['allowed']:
                return {
                    'compatible': False,
                    'reason': 'Field removals not allowed'
                }
            
            # Check field type changes
            for type_change in changes.get('field_type_changes', []):
                old_type = type_change['old_type']
                new_type = type_change['new_type']
                
                compatible_types = self.evolution_rules['field_type_changes']['compatible_types']
                if old_type in compatible_types:
                    if new_type not in compatible_types[old_type]:
                        return {
                            'compatible': False,
                            'reason': f'Incompatible type change: {old_type} -> {new_type}'
                        }
                else:
                    return {
                        'compatible': False,
                        'reason': f'Unknown type: {old_type}'
                    }
            
            # Check constraint changes
            for constraint_change in changes.get('constraint_changes', []):
                old_constraints = constraint_change['old_constraints']
                new_constraints = constraint_change['new_constraints']
                
                # Check if constraints are being tightened (not allowed)
                if not self.evolution_rules['constraint_changes']['tightening_allowed']:
                    for constraint_type in ['min', 'max']:
                        if constraint_type in old_constraints and constraint_type in new_constraints:
                            if constraint_type == 'min' and new_constraints[constraint_type] > old_constraints[constraint_type]:
                                return {
                                    'compatible': False,
                                    'reason': f'Constraint tightening not allowed: {constraint_type}'
                                }
                            elif constraint_type == 'max' and new_constraints[constraint_type] < old_constraints[constraint_type]:
                                return {
                                    'compatible': False,
                                    'reason': f'Constraint tightening not allowed: {constraint_type}'
                                }
            
            return {'compatible': True, 'reason': 'Schema evolution is compatible'}
            
        except Exception as e:
            logger.error(f"Error checking evolution compatibility: {e}")
            return {'compatible': False, 'reason': str(e)}
    
    def _generate_evolution_plan(self, changes: Dict[str, Any], schema_name: str, 
                               evolution_type: str) -> Dict[str, Any]:
        """Generate evolution plan."""
        try:
            schema_config = self.evolution_config.get('schemas', {}).get(schema_name, {})
            evolution_strategy = schema_config.get('evolution_strategy', 'gradual')
            
            plan = {
                'strategy': evolution_strategy,
                'type': evolution_type,
                'steps': [],
                'estimated_duration': 0,
                'rollback_plan': {}
            }
            
            # Generate steps based on changes
            if changes.get('field_additions'):
                plan['steps'].append({
                    'type': 'field_addition',
                    'description': 'Add new fields to schema',
                    'fields': changes['field_additions'],
                    'duration_minutes': 5
                })
            
            if changes.get('field_removals'):
                plan['steps'].append({
                    'type': 'field_removal',
                    'description': 'Remove deprecated fields',
                    'fields': changes['field_removals'],
                    'duration_minutes': 10
                })
            
            if changes.get('field_type_changes'):
                plan['steps'].append({
                    'type': 'field_type_change',
                    'description': 'Update field types',
                    'changes': changes['field_type_changes'],
                    'duration_minutes': 15
                })
            
            if changes.get('constraint_changes'):
                plan['steps'].append({
                    'type': 'constraint_change',
                    'description': 'Update field constraints',
                    'changes': changes['constraint_changes'],
                    'duration_minutes': 10
                })
            
            if changes.get('format_changes'):
                plan['steps'].append({
                    'type': 'format_change',
                    'description': 'Update field formats',
                    'changes': changes['format_changes'],
                    'duration_minutes': 5
                })
            
            # Calculate estimated duration
            plan['estimated_duration'] = sum(step['duration_minutes'] for step in plan['steps'])
            
            # Generate rollback plan
            plan['rollback_plan'] = self._generate_rollback_plan(changes)
            
            return plan
            
        except Exception as e:
            logger.error(f"Error generating evolution plan: {e}")
            return {}
    
    def _execute_evolution(self, schema_name: str, current_schema: Dict[str, Any], 
                         new_schema: Dict[str, Any], evolution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute schema evolution."""
        try:
            execution_result = {
                'status': 'success',
                'steps_completed': [],
                'steps_failed': [],
                'execution_time_minutes': 0,
                'start_time': datetime.now().isoformat()
            }
            
            start_time = datetime.now()
            
            # Execute each step
            for step in evolution_plan.get('steps', []):
                try:
                    step_result = self._execute_evolution_step(schema_name, step, current_schema, new_schema)
                    
                    if step_result['success']:
                        execution_result['steps_completed'].append(step)
                    else:
                        execution_result['steps_failed'].append({
                            'step': step,
                            'error': step_result['error']
                        })
                        
                except Exception as e:
                    logger.error(f"Error executing evolution step: {e}")
                    execution_result['steps_failed'].append({
                        'step': step,
                        'error': str(e)
                    })
            
            # Calculate execution time
            end_time = datetime.now()
            execution_result['execution_time_minutes'] = (end_time - start_time).total_seconds() / 60
            execution_result['end_time'] = end_time.isoformat()
            
            # Determine overall status
            if execution_result['steps_failed']:
                execution_result['status'] = 'partial_success' if execution_result['steps_completed'] else 'failed'
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Error executing evolution: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _execute_evolution_step(self, schema_name: str, step: Dict[str, Any], 
                              current_schema: Dict[str, Any], new_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single evolution step."""
        try:
            step_type = step['type']
            
            if step_type == 'field_addition':
                return self._execute_field_addition(schema_name, step, new_schema)
            elif step_type == 'field_removal':
                return self._execute_field_removal(schema_name, step, current_schema)
            elif step_type == 'field_type_change':
                return self._execute_field_type_change(schema_name, step, new_schema)
            elif step_type == 'constraint_change':
                return self._execute_constraint_change(schema_name, step, new_schema)
            elif step_type == 'format_change':
                return self._execute_format_change(schema_name, step, new_schema)
            else:
                return {'success': False, 'error': f'Unknown step type: {step_type}'}
            
        except Exception as e:
            logger.error(f"Error executing evolution step: {e}")
            return {'success': False, 'error': str(e)}
    
    def _execute_field_addition(self, schema_name: str, step: Dict[str, Any], 
                              new_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Execute field addition step."""
        try:
            # In a real implementation, this would update the actual schema storage
            logger.info(f"Adding fields to {schema_name}: {step['fields']}")
            return {'success': True, 'message': 'Fields added successfully'}
            
        except Exception as e:
            logger.error(f"Error adding fields: {e}")
            return {'success': False, 'error': str(e)}
    
    def _execute_field_removal(self, schema_name: str, step: Dict[str, Any], 
                             current_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Execute field removal step."""
        try:
            # In a real implementation, this would update the actual schema storage
            logger.info(f"Removing fields from {schema_name}: {step['fields']}")
            return {'success': True, 'message': 'Fields removed successfully'}
            
        except Exception as e:
            logger.error(f"Error removing fields: {e}")
            return {'success': False, 'error': str(e)}
    
    def _execute_field_type_change(self, schema_name: str, step: Dict[str, Any], 
                                 new_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Execute field type change step."""
        try:
            # In a real implementation, this would update the actual schema storage
            logger.info(f"Changing field types in {schema_name}: {step['changes']}")
            return {'success': True, 'message': 'Field types changed successfully'}
            
        except Exception as e:
            logger.error(f"Error changing field types: {e}")
            return {'success': False, 'error': str(e)}
    
    def _execute_constraint_change(self, schema_name: str, step: Dict[str, Any], 
                                 new_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Execute constraint change step."""
        try:
            # In a real implementation, this would update the actual schema storage
            logger.info(f"Changing constraints in {schema_name}: {step['changes']}")
            return {'success': True, 'message': 'Constraints changed successfully'}
            
        except Exception as e:
            logger.error(f"Error changing constraints: {e}")
            return {'success': False, 'error': str(e)}
    
    def _execute_format_change(self, schema_name: str, step: Dict[str, Any], 
                             new_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Execute format change step."""
        try:
            # In a real implementation, this would update the actual schema storage
            logger.info(f"Changing formats in {schema_name}: {step['changes']}")
            return {'success': True, 'message': 'Formats changed successfully'}
            
        except Exception as e:
            logger.error(f"Error changing formats: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_rollback_plan(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        """Generate rollback plan for schema evolution."""
        try:
            rollback_plan = {
                'steps': [],
                'estimated_duration': 0
            }
            
            # Generate rollback steps (reverse of evolution steps)
            if changes.get('field_additions'):
                rollback_plan['steps'].append({
                    'type': 'field_removal',
                    'description': 'Remove added fields',
                    'fields': changes['field_additions'],
                    'duration_minutes': 5
                })
            
            if changes.get('field_removals'):
                rollback_plan['steps'].append({
                    'type': 'field_addition',
                    'description': 'Restore removed fields',
                    'fields': changes['field_removals'],
                    'duration_minutes': 10
                })
            
            if changes.get('field_type_changes'):
                rollback_plan['steps'].append({
                    'type': 'field_type_change',
                    'description': 'Restore original field types',
                    'changes': [{'field': change['field'], 'old_type': change['new_type'], 'new_type': change['old_type']} 
                               for change in changes['field_type_changes']],
                    'duration_minutes': 15
                })
            
            if changes.get('constraint_changes'):
                rollback_plan['steps'].append({
                    'type': 'constraint_change',
                    'description': 'Restore original constraints',
                    'changes': [{'field': change['field'], 'old_constraints': change['new_constraints'], 'new_constraints': change['old_constraints']} 
                               for change in changes['constraint_changes']],
                    'duration_minutes': 10
                })
            
            if changes.get('format_changes'):
                rollback_plan['steps'].append({
                    'type': 'format_change',
                    'description': 'Restore original formats',
                    'changes': [{'field': change['field'], 'old_format': change['new_format'], 'new_format': change['old_format']} 
                               for change in changes['format_changes']],
                    'duration_minutes': 5
                })
            
            # Calculate estimated duration
            rollback_plan['estimated_duration'] = sum(step['duration_minutes'] for step in rollback_plan['steps'])
            
            return rollback_plan
            
        except Exception as e:
            logger.error(f"Error generating rollback plan: {e}")
            return {}
    
    def _update_evolution_history(self, schema_name: str, changes: Dict[str, Any], 
                                evolution_result: Dict[str, Any]) -> None:
        """Update evolution history."""
        try:
            if schema_name not in self.evolution_history:
                self.evolution_history[schema_name] = []
            
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'changes': changes,
                'result': evolution_result,
                'status': evolution_result.get('status', 'unknown')
            }
            
            self.evolution_history[schema_name].append(history_entry)
            
            # Keep only last 100 history entries
            if len(self.evolution_history[schema_name]) > 100:
                self.evolution_history[schema_name] = self.evolution_history[schema_name][-100:]
                
        except Exception as e:
            logger.error(f"Error updating evolution history: {e}")
    
    def _is_evolution_enabled(self, schema_name: str) -> bool:
        """Check if evolution is enabled for a schema."""
        try:
            if not self.evolution_config.get('enabled', True):
                return False
            
            schema_config = self.evolution_config.get('schemas', {}).get(schema_name, {})
            return schema_config.get('enabled', True)
        except Exception as e:
            logger.error(f"Error checking evolution enablement: {e}")
            return False
    
    def get_evolution_history(self, schema_name: str) -> List[Dict[str, Any]]:
        """Get evolution history for a schema."""
        try:
            return self.evolution_history.get(schema_name, [])
        except Exception as e:
            logger.error(f"Error getting evolution history: {e}")
            return []
    
    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get evolution statistics."""
        try:
            stats = {
                'configuration': self.evolution_config.copy(),
                'evolution_rules': self.evolution_rules.copy(),
                'schema_histories': {}
            }
            
            for schema_name, history in self.evolution_history.items():
                stats['schema_histories'][schema_name] = {
                    'total_evolutions': len(history),
                    'successful_evolutions': len([h for h in history if h['status'] == 'success']),
                    'failed_evolutions': len([h for h in history if h['status'] == 'failed']),
                    'last_evolution': history[-1] if history else None
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting evolution statistics: {e}")
            return {}
    
    def rollback_evolution(self, schema_name: str, evolution_id: str) -> Dict[str, Any]:
        """Rollback a specific evolution."""
        try:
            if not self.evolution_config.get('rollback_enabled', True):
                return {'status': 'disabled', 'message': 'Rollback is disabled'}
            
            if schema_name not in self.evolution_history:
                return {'status': 'error', 'error': f'No evolution history found for {schema_name}'}
            
            # Find the evolution to rollback
            evolution_to_rollback = None
            for evolution in self.evolution_history[schema_name]:
                if evolution.get('timestamp') == evolution_id:
                    evolution_to_rollback = evolution
                    break
            
            if not evolution_to_rollback:
                return {'status': 'error', 'error': f'Evolution not found: {evolution_id}'}
            
            # Generate rollback plan
            rollback_plan = self._generate_rollback_plan(evolution_to_rollback['changes'])
            
            # Execute rollback
            rollback_result = self._execute_rollback(schema_name, rollback_plan)
            
            result = {
                'status': 'success',
                'schema_name': schema_name,
                'evolution_id': evolution_id,
                'rollback_plan': rollback_plan,
                'rollback_result': rollback_result,
                'rolled_back_at': datetime.now().isoformat()
            }
            
            logger.info(f"Schema evolution rolled back for {schema_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error rolling back evolution: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _execute_rollback(self, schema_name: str, rollback_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute rollback plan."""
        try:
            rollback_result = {
                'status': 'success',
                'steps_completed': [],
                'steps_failed': [],
                'execution_time_minutes': 0,
                'start_time': datetime.now().isoformat()
            }
            
            start_time = datetime.now()
            
            # Execute each rollback step
            for step in rollback_plan.get('steps', []):
                try:
                    step_result = self._execute_rollback_step(schema_name, step)
                    
                    if step_result['success']:
                        rollback_result['steps_completed'].append(step)
                    else:
                        rollback_result['steps_failed'].append({
                            'step': step,
                            'error': step_result['error']
                        })
                        
                except Exception as e:
                    logger.error(f"Error executing rollback step: {e}")
                    rollback_result['steps_failed'].append({
                        'step': step,
                        'error': str(e)
                    })
            
            # Calculate execution time
            end_time = datetime.now()
            rollback_result['execution_time_minutes'] = (end_time - start_time).total_seconds() / 60
            rollback_result['end_time'] = end_time.isoformat()
            
            # Determine overall status
            if rollback_result['steps_failed']:
                rollback_result['status'] = 'partial_success' if rollback_result['steps_completed'] else 'failed'
            
            return rollback_result
            
        except Exception as e:
            logger.error(f"Error executing rollback: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _execute_rollback_step(self, schema_name: str, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single rollback step."""
        try:
            step_type = step['type']
            
            if step_type == 'field_addition':
                return self._execute_field_addition(schema_name, step, {})
            elif step_type == 'field_removal':
                return self._execute_field_removal(schema_name, step, {})
            elif step_type == 'field_type_change':
                return self._execute_field_type_change(schema_name, step, {})
            elif step_type == 'constraint_change':
                return self._execute_constraint_change(schema_name, step, {})
            elif step_type == 'format_change':
                return self._execute_format_change(schema_name, step, {})
            else:
                return {'success': False, 'error': f'Unknown rollback step type: {step_type}'}
            
        except Exception as e:
            logger.error(f"Error executing rollback step: {e}")
            return {'success': False, 'error': str(e)}
