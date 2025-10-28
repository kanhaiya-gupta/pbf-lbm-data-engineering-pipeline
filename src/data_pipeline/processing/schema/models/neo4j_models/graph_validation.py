"""
Neo4j Graph Validation Engine

This module provides comprehensive validation for Neo4j knowledge graph data.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timezone
from pydantic import BaseModel, ValidationError as PydanticValidationError, Field
import logging

from .core_graph_models import (
    ProcessModel, MachineModel, PartModel, BuildModel, MaterialModel,
    QualityModel, SensorModel, UserModel, OperatorModel, AlertModel,
    DefectModel, ImageModel, LogModel, InspectionModel,
    # New node models
    ThermalImageModel, ProcessImageModel, CTScanImageModel, PowderBedImageModel,
    BuildFileModel, ModelFileModel, LogFileModel,
    ProcessCacheModel, AnalyticsCacheModel, JobQueueModel, UserSessionModel,
    SensorReadingModel, ProcessMonitoringModel, MachineStatusModel, AlertEventModel,
    BatchModel, MeasurementModel, MachineConfigModel, SensorTypeModel
)
from .relationship_models import (
    ProcessMachineRelationship, ProcessPartRelationship, ProcessBuildRelationship,
    ProcessMaterialRelationship, ProcessQualityRelationship, ProcessSensorRelationship,
    ProcessOperatorRelationship, ProcessAlertRelationship, ProcessDefectRelationship,
    ProcessImageRelationship, ProcessLogRelationship
)

logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION RESULT MODELS
# =============================================================================

class ValidationError(BaseModel):
    """Validation error model."""
    field: str = Field(..., description="Field name")
    message: str = Field(..., description="Error message")
    value: Any = Field(..., description="Invalid value")
    error_type: str = Field(..., description="Error type")
    
    def __init__(self, field: str, message: str, value: Any, error_type: str, **kwargs):
        super().__init__(field=field, message=message, value=value, error_type=error_type, **kwargs)

class ValidationWarning(BaseModel):
    """Validation warning model."""
    field: str = Field(..., description="Field name")
    message: str = Field(..., description="Warning message")
    value: Any = Field(..., description="Value")
    suggestion: Optional[str] = Field(None, description="Suggestion")

class NodeValidationResult(BaseModel):
    """Node validation result model."""
    node_id: str = Field(..., description="Node ID")
    node_type: str = Field(..., description="Node type")
    valid: bool = Field(..., description="Validation status")
    errors: List[ValidationError] = Field(default_factory=list, description="Validation errors")
    warnings: List[ValidationWarning] = Field(default_factory=list, description="Validation warnings")
    validated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Validation timestamp")

class RelationshipValidationResult(BaseModel):
    """Relationship validation result model."""
    relationship_id: str = Field(..., description="Relationship ID")
    relationship_type: str = Field(..., description="Relationship type")
    from_id: str = Field(..., description="Source node ID")
    to_id: str = Field(..., description="Target node ID")
    valid: bool = Field(..., description="Validation status")
    errors: List[ValidationError] = Field(default_factory=list, description="Validation errors")
    warnings: List[ValidationWarning] = Field(default_factory=list, description="Validation warnings")
    validated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Validation timestamp")

class GraphValidationResult(BaseModel):
    """Graph validation result model."""
    total_nodes: int = Field(..., description="Total number of nodes")
    total_relationships: int = Field(..., description="Total number of relationships")
    valid_nodes: int = Field(..., description="Number of valid nodes")
    invalid_nodes: int = Field(..., description="Number of invalid nodes")
    valid_relationships: int = Field(..., description="Number of valid relationships")
    invalid_relationships: int = Field(..., description="Number of invalid relationships")
    node_results: List[NodeValidationResult] = Field(default_factory=list, description="Node validation results")
    relationship_results: List[RelationshipValidationResult] = Field(default_factory=list, description="Relationship validation results")
    validation_time: float = Field(..., description="Validation time in seconds")
    validated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Validation timestamp")


# =============================================================================
# VALIDATION ENGINE
# =============================================================================

class GraphValidationEngine:
    """Graph validation engine for Neo4j knowledge graph."""
    
    def __init__(self):
        """Initialize the validation engine."""
        self.node_models = {
            'Process': ProcessModel,
            'Machine': MachineModel,
            'Part': PartModel,
            'Build': BuildModel,
            'Material': MaterialModel,
            'Quality': QualityModel,
            'Sensor': SensorModel,
            'User': UserModel,
            'Operator': OperatorModel,
            'Alert': AlertModel,
            'Defect': DefectModel,
            'Image': ImageModel,
            'Log': LogModel,
            'Inspection': InspectionModel,
            # New node types
            'ThermalImage': ThermalImageModel,
            'ProcessImage': ProcessImageModel,
            'CTScanImage': CTScanImageModel,
            'PowderBedImage': PowderBedImageModel,
            'BuildFile': BuildFileModel,
            'ModelFile': ModelFileModel,
            'LogFile': LogFileModel,
            'ProcessCache': ProcessCacheModel,
            'AnalyticsCache': AnalyticsCacheModel,
            'JobQueue': JobQueueModel,
            'UserSession': UserSessionModel,
            'SensorReading': SensorReadingModel,
            'ProcessMonitoring': ProcessMonitoringModel,
            'MachineStatus': MachineStatusModel,
            'AlertEvent': AlertEventModel,
            'Batch': BatchModel,
            'Measurement': MeasurementModel,
            'MachineConfig': MachineConfigModel,
            'SensorType': SensorTypeModel
        }
        
        self.relationship_models = {
            'USES_MACHINE': ProcessMachineRelationship,
            'CREATES_PART': ProcessPartRelationship,
            'PART_OF_BUILD': ProcessBuildRelationship,
            'USES_MATERIAL': ProcessMaterialRelationship,
            'HAS_QUALITY': ProcessQualityRelationship,
            'MONITORED_BY': ProcessSensorRelationship,
            'OPERATED_BY': ProcessOperatorRelationship,
            'GENERATES_ALERT': ProcessAlertRelationship,
            'HAS_DEFECT': ProcessDefectRelationship,
            'CAPTURED_BY': ProcessImageRelationship,
            'LOGGED_IN': ProcessLogRelationship
        }
    
    def validate_node(self, node_data: Dict[str, Any], node_type: str) -> NodeValidationResult:
        """
        Validate a single node.
        
        Args:
            node_data: Node data dictionary
            node_type: Node type string
            
        Returns:
            NodeValidationResult: Validation result
        """
        node_id = node_data.get('graph_id', node_data.get('id', 'unknown'))
        errors = []
        warnings = []
        
        try:
            # Get the appropriate model class
            model_class = self.node_models.get(node_type)
            if not model_class:
                errors.append(ValidationError(
                    field='node_type',
                    message=f'Unknown node type: {node_type}',
                    value=node_type,
                    error_type='UnknownNodeType'
                ))
                return NodeValidationResult(
                    node_id=node_id,
                    node_type=node_type,
                    valid=False,
                    errors=errors,
                    warnings=warnings
                )
            
            # Validate the node data
            model_instance = model_class(**node_data)
            
            # Additional business logic validations
            self._validate_node_business_logic(model_instance, node_type, warnings)
            
            return NodeValidationResult(
                node_id=node_id,
                node_type=node_type,
                valid=True,
                errors=errors,
                warnings=warnings
            )
            
        except PydanticValidationError as e:
            for error in e.errors():
                errors.append(ValidationError(
                    field=error.get('loc', ['unknown'])[0],
                    message=error.get('msg', 'Validation error'),
                    value=error.get('input', 'unknown'),
                    error_type=error.get('type', 'ValidationError')
                ))
            
            return NodeValidationResult(
                node_id=node_id,
                node_type=node_type,
                valid=False,
                errors=errors,
                warnings=warnings
            )
        except Exception as e:
            errors.append(ValidationError(
                field='general',
                message=f'Unexpected validation error: {str(e)}',
                value=node_data,
                error_type='UnexpectedError'
            ))
            
            return NodeValidationResult(
                node_id=node_id,
                node_type=node_type,
                valid=False,
                errors=errors,
                warnings=warnings
            )
    
    def validate_relationship(self, relationship_data: Dict[str, Any], relationship_type: str) -> RelationshipValidationResult:
        """
        Validate a single relationship.
        
        Args:
            relationship_data: Relationship data dictionary
            relationship_type: Relationship type string
            
        Returns:
            RelationshipValidationResult: Validation result
        """
        relationship_id = relationship_data.get('graph_id', 'unknown')
        from_id = relationship_data.get('from_id', 'unknown')
        to_id = relationship_data.get('to_id', 'unknown')
        errors = []
        warnings = []
        
        try:
            # Get the appropriate model class
            model_class = self.relationship_models.get(relationship_type)
            if not model_class:
                errors.append(ValidationError(
                    field='relationship_type',
                    message=f'Unknown relationship type: {relationship_type}',
                    value=relationship_type,
                    error_type='UnknownRelationshipType'
                ))
                return RelationshipValidationResult(
                    relationship_id=relationship_id,
                    relationship_type=relationship_type,
                    from_id=from_id,
                    to_id=to_id,
                    valid=False,
                    errors=errors,
                    warnings=warnings
                )
            
            # Validate the relationship data
            model_instance = model_class(**relationship_data)
            
            # Additional business logic validations
            self._validate_relationship_business_logic(model_instance, relationship_type, warnings)
            
            return RelationshipValidationResult(
                relationship_id=relationship_id,
                relationship_type=relationship_type,
                from_id=from_id,
                to_id=to_id,
                valid=True,
                errors=errors,
                warnings=warnings
            )
            
        except PydanticValidationError as e:
            for error in e.errors():
                errors.append(ValidationError(
                    field=error.get('loc', ['unknown'])[0],
                    message=error.get('msg', 'Validation error'),
                    value=error.get('input', 'unknown'),
                    error_type=error.get('type', 'ValidationError')
                ))
            
            return RelationshipValidationResult(
                relationship_id=relationship_id,
                relationship_type=relationship_type,
                from_id=from_id,
                to_id=to_id,
                valid=False,
                errors=errors,
                warnings=warnings
            )
        except Exception as e:
            errors.append(ValidationError(
                field='general',
                message=f'Unexpected validation error: {str(e)}',
                value=relationship_data,
                error_type='UnexpectedError'
            ))
            
            return RelationshipValidationResult(
                relationship_id=relationship_id,
                relationship_type=relationship_type,
                from_id=from_id,
                to_id=to_id,
                valid=False,
                errors=errors,
                warnings=warnings
            )
    
    def validate_graph(self, nodes: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> GraphValidationResult:
        """
        Validate a complete graph.
        
        Args:
            nodes: List of node data dictionaries
            relationships: List of relationship data dictionaries
            
        Returns:
            GraphValidationResult: Complete validation result
        """
        start_time = datetime.now(timezone.utc)
        
        # Validate nodes
        node_results = []
        for node in nodes:
            node_type = node.get('node_type', 'Unknown')
            result = self.validate_node(node, node_type)
            node_results.append(result)
        
        # Validate relationships
        relationship_results = []
        for relationship in relationships:
            relationship_type = relationship.get('relationship_type', 'Unknown')
            result = self.validate_relationship(relationship, relationship_type)
            relationship_results.append(result)
        
        # Calculate statistics
        valid_nodes = sum(1 for r in node_results if r.valid)
        invalid_nodes = len(node_results) - valid_nodes
        valid_relationships = sum(1 for r in relationship_results if r.valid)
        invalid_relationships = len(relationship_results) - valid_relationships
        
        # Calculate validation time
        validation_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        return GraphValidationResult(
            total_nodes=len(nodes),
            total_relationships=len(relationships),
            valid_nodes=valid_nodes,
            invalid_nodes=invalid_nodes,
            valid_relationships=valid_relationships,
            invalid_relationships=invalid_relationships,
            node_results=node_results,
            relationship_results=relationship_results,
            validation_time=validation_time
        )
    
    def _validate_node_business_logic(self, model_instance: BaseModel, node_type: str, warnings: List[ValidationWarning]):
        """Validate node business logic."""
        if node_type == 'Process':
            self._validate_process_business_logic(model_instance, warnings)
        elif node_type == 'Machine':
            self._validate_machine_business_logic(model_instance, warnings)
        elif node_type == 'Part':
            self._validate_part_business_logic(model_instance, warnings)
        elif node_type == 'Build':
            self._validate_build_business_logic(model_instance, warnings)
        elif node_type == 'Quality':
            self._validate_quality_business_logic(model_instance, warnings)
        elif node_type == 'Alert':
            self._validate_alert_business_logic(model_instance, warnings)
    
    def _validate_process_business_logic(self, process: ProcessModel, warnings: List[ValidationWarning]):
        """Validate process business logic."""
        # Check if laser power is within reasonable range for material
        if process.material_type == 'Ti6Al4V' and process.laser_power > 300:
            warnings.append(ValidationWarning(
                field='laser_power',
                message='Laser power seems high for Ti6Al4V material',
                value=process.laser_power,
                suggestion='Consider reducing laser power for better quality'
            ))
        
        # Check if scan speed is appropriate for layer thickness
        if process.layer_thickness > 0.05 and process.scan_speed > 1500:
            warnings.append(ValidationWarning(
                field='scan_speed',
                message='Scan speed may be too high for thick layers',
                value=process.scan_speed,
                suggestion='Consider reducing scan speed for better layer adhesion'
            ))
        
        # Check if density is reasonable
        if process.density is not None and process.density < 0.9:
            warnings.append(ValidationWarning(
                field='density',
                message='Process density is low, may indicate quality issues',
                value=process.density,
                suggestion='Check process parameters and material quality'
            ))
    
    def _validate_machine_business_logic(self, machine: MachineModel, warnings: List[ValidationWarning]):
        """Validate machine business logic."""
        # Check if machine is overdue for maintenance
        if machine.maintenance_date and machine.utilization_rate and machine.utilization_rate > 0.8:
            days_since_maintenance = (datetime.now().date() - machine.maintenance_date).days
            if days_since_maintenance > 30:
                warnings.append(ValidationWarning(
                    field='maintenance_date',
                    message='Machine may be overdue for maintenance',
                    value=machine.maintenance_date,
                    suggestion='Schedule maintenance soon'
                ))
        
        # Check if utilization rate is too high
        if machine.utilization_rate and machine.utilization_rate > 0.95:
            warnings.append(ValidationWarning(
                field='utilization_rate',
                message='Machine utilization is very high',
                value=machine.utilization_rate,
                suggestion='Consider load balancing or adding capacity'
            ))
    
    def _validate_part_business_logic(self, part: PartModel, warnings: List[ValidationWarning]):
        """Validate part business logic."""
        # Check if part dimensions are reasonable
        if part.dimensions:
            max_dimension = max(part.dimensions.x, part.dimensions.y, part.dimensions.z)
            if max_dimension > 500:
                warnings.append(ValidationWarning(
                    field='dimensions',
                    message='Part dimensions are very large',
                    value=part.dimensions.dict(),
                    suggestion='Consider splitting into smaller parts'
                ))
        
        # Check if weight is reasonable for dimensions
        if part.weight and part.volume and part.material_type:
            calculated_density = part.weight / (part.volume / 1000)  # Convert cm³ to m³
            if part.material_type == 'Ti6Al4V' and calculated_density < 4.0:
                warnings.append(ValidationWarning(
                    field='weight',
                    message='Part weight seems low for Ti6Al4V material',
                    value=part.weight,
                    suggestion='Check weight measurement or material density'
                ))
    
    def _validate_build_business_logic(self, build: BuildModel, warnings: List[ValidationWarning]):
        """Validate build business logic."""
        # Check if build duration is reasonable
        if build.total_duration and build.total_parts:
            avg_duration_per_part = build.total_duration / build.total_parts
            if avg_duration_per_part < 3600:  # Less than 1 hour per part
                warnings.append(ValidationWarning(
                    field='total_duration',
                    message='Build duration seems short for number of parts',
                    value=build.total_duration,
                    suggestion='Check if duration includes all processing steps'
                ))
        
        # Check if success rate is too low
        if build.success_rate and build.success_rate < 0.8:
            warnings.append(ValidationWarning(
                field='success_rate',
                message='Build success rate is low',
                value=build.success_rate,
                suggestion='Review process parameters and quality control'
            ))
    
    def _validate_quality_business_logic(self, quality: QualityModel, warnings: List[ValidationWarning]):
        """Validate quality business logic."""
        # Check if quality metrics are consistent
        if quality.metrics:
            if quality.metrics.density and quality.metrics.density < 0.95:
                warnings.append(ValidationWarning(
                    field='metrics.density',
                    message='Quality density is low',
                    value=quality.metrics.density,
                    suggestion='Check process parameters for better density'
                ))
            
            if quality.metrics.surface_roughness and quality.metrics.surface_roughness > 10:
                warnings.append(ValidationWarning(
                    field='metrics.surface_roughness',
                    message='Surface roughness is high',
                    value=quality.metrics.surface_roughness,
                    suggestion='Consider post-processing or parameter optimization'
                ))
    
    def _validate_alert_business_logic(self, alert: AlertModel, warnings: List[ValidationWarning]):
        """Validate alert business logic."""
        # Check if alert is old and unresolved
        if alert.status == 'active' and alert.timestamp:
            hours_since_alert = (datetime.now(timezone.utc) - alert.timestamp).total_seconds() / 3600
            if hours_since_alert > 24:
                warnings.append(ValidationWarning(
                    field='status',
                    message='Alert has been active for more than 24 hours',
                    value=alert.status,
                    suggestion='Consider escalating or resolving the alert'
                ))
        
        # Check if critical alerts have resolution time
        if alert.severity == 'critical' and alert.resolution_time is None:
            warnings.append(ValidationWarning(
                field='resolution_time',
                message='Critical alert has no resolution time',
                value=alert.resolution_time,
                suggestion='Set resolution time for critical alerts'
            ))
    
    def _validate_relationship_business_logic(self, relationship: BaseModel, relationship_type: str, warnings: List[ValidationWarning]):
        """Validate relationship business logic."""
        if relationship_type == 'USES_MACHINE':
            self._validate_machine_usage_relationship(relationship, warnings)
        elif relationship_type == 'CREATES_PART':
            self._validate_part_creation_relationship(relationship, warnings)
        elif relationship_type == 'HAS_QUALITY':
            self._validate_quality_relationship(relationship, warnings)
    
    def _validate_machine_usage_relationship(self, relationship: ProcessMachineRelationship, warnings: List[ValidationWarning]):
        """Validate machine usage relationship."""
        # Check if duration is reasonable
        if relationship.duration and relationship.duration > 86400:  # More than 24 hours
            warnings.append(ValidationWarning(
                field='duration',
                message='Process duration is very long',
                value=relationship.duration,
                suggestion='Check if duration includes all processing steps'
            ))
        
        # Check if utilization is reasonable
        if relationship.utilization and relationship.utilization > 1.0:
            warnings.append(ValidationWarning(
                field='utilization',
                message='Machine utilization exceeds 100%',
                value=relationship.utilization,
                suggestion='Check utilization calculation'
            ))
    
    def _validate_part_creation_relationship(self, relationship: ProcessPartRelationship, warnings: List[ValidationWarning]):
        """Validate part creation relationship."""
        # Check if success rate is reasonable
        if relationship.success_rate and relationship.success_rate < 0.5:
            warnings.append(ValidationWarning(
                field='success_rate',
                message='Part creation success rate is very low',
                value=relationship.success_rate,
                suggestion='Review process parameters and quality control'
            ))
        
        # Check if quantity is reasonable
        if relationship.quantity and relationship.quantity > 100:
            warnings.append(ValidationWarning(
                field='quantity',
                message='Part quantity is very high for single process',
                value=relationship.quantity,
                suggestion='Consider splitting into multiple processes'
            ))
    
    def _validate_quality_relationship(self, relationship: ProcessQualityRelationship, warnings: List[ValidationWarning]):
        """Validate quality relationship."""
        # Check if confidence is reasonable
        if relationship.confidence and relationship.confidence < 0.7:
            warnings.append(ValidationWarning(
                field='confidence',
                message='Quality measurement confidence is low',
                value=relationship.confidence,
                suggestion='Review measurement methods and equipment'
            ))
        
        # Check if correlation is reasonable
        if relationship.correlation and abs(relationship.correlation) > 1.0:
            warnings.append(ValidationWarning(
                field='correlation',
                message='Quality correlation is outside valid range',
                value=relationship.correlation,
                suggestion='Check correlation calculation'
            ))
    
    def get_validation_summary(self, result: GraphValidationResult) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            'total_nodes': result.total_nodes,
            'total_relationships': result.total_relationships,
            'valid_nodes': result.valid_nodes,
            'invalid_nodes': result.invalid_nodes,
            'valid_relationships': result.valid_relationships,
            'invalid_relationships': result.invalid_relationships,
            'validation_time': result.validation_time,
            'node_validation_rate': result.valid_nodes / result.total_nodes if result.total_nodes > 0 else 0,
            'relationship_validation_rate': result.valid_relationships / result.total_relationships if result.total_relationships > 0 else 0,
            'total_errors': sum(len(r.errors) for r in result.node_results + result.relationship_results),
            'total_warnings': sum(len(r.warnings) for r in result.node_results + result.relationship_results)
        }
