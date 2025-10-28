"""
Elasticsearch Document Models for PBF-LB/M Data Pipeline

This module provides Pydantic models that EXACTLY match the Elasticsearch JSON schemas,
ensuring perfect validation and consistency between models and Elasticsearch indices.
"""

from typing import Any, Dict, Optional, List, Union, Literal
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


class ProcessStatus(str, Enum):
    """Process status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MachineStatus(str, Enum):
    """Machine status enumeration."""
    IDLE = "idle"
    RUNNING = "running"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    OFFLINE = "offline"


class SensorType(str, Enum):
    """Sensor type enumeration."""
    THERMAL = "THERMAL"
    OPTICAL = "OPTICAL"
    ACOUSTIC = "ACOUSTIC"
    VIBRATION = "VIBRATION"
    PRESSURE = "PRESSURE"
    GAS_ANALYSIS = "GAS_ANALYSIS"
    MELT_POOL = "MELT_POOL"
    LAYER_HEIGHT = "LAYER_HEIGHT"


class MaterialType(str, Enum):
    """Material type enumeration."""
    TITANIUM = "TITANIUM"
    ALUMINUM = "ALUMINUM"
    STEEL = "STEEL"
    INCONEL = "INCONEL"
    COBALT_CHROME = "COBALT_CHROME"
    NICKEL = "NICKEL"


class QualityStatus(str, Enum):
    """Quality status enumeration."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    REJECTED = "rejected"


class AlertSeverity(str, Enum):
    """Alert severity enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BaseElasticsearchDocument(BaseModel):
    """
    Base Elasticsearch document model that matches JSON schemas exactly.
    """
    
    # Elasticsearch-specific fields
    id: Optional[str] = Field(None, description="Elasticsearch document ID")
    index: Optional[str] = Field(None, description="Elasticsearch index name")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Document creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Document last update timestamp")
    
    # Document identification
    document_type: str = Field(..., description="Type of document")
    document_id: str = Field(..., pattern="^[A-Z0-9_]+$", description="Unique document identifier")
    
    # PostgreSQL relationships
    process_id: Optional[str] = Field(None, pattern="^[A-Z0-9_]+$", description="Associated PostgreSQL process ID")
    build_id: Optional[str] = Field(None, pattern="^[A-Z0-9_]+$", description="Associated PostgreSQL build ID")
    part_id: Optional[str] = Field(None, pattern="^[A-Z0-9_]+$", description="Associated PostgreSQL part ID")
    machine_id: Optional[str] = Field(None, pattern="^[A-Z0-9_]+$", description="Associated PostgreSQL machine ID")
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    
    @model_validator(mode='after')
    def validate_document(self):
        """Validate document structure and relationships."""
        if not self.document_id:
            raise ValueError("document_id is required")
        return self


class ProcessParameters(BaseModel):
    """Process parameters for PBF manufacturing."""
    laser_power: float = Field(..., ge=0, le=1000, description="Laser power in watts")
    scan_speed: float = Field(..., ge=0, le=10000, description="Scan speed in mm/s")
    layer_thickness: float = Field(..., ge=0.01, le=1.0, description="Layer thickness in mm")
    hatch_spacing: float = Field(..., ge=0.01, le=1.0, description="Hatch spacing in mm")
    build_plate_temperature: float = Field(..., ge=0, le=500, description="Build plate temperature in °C")
    exposure_time: Optional[float] = Field(None, ge=0, le=1000, description="Exposure time in ms")
    focus_offset: Optional[float] = Field(None, ge=-10, le=10, description="Focus offset in mm")


class MaterialInfo(BaseModel):
    """Material information for PBF manufacturing."""
    material_type: MaterialType = Field(..., description="Type of material")
    powder_batch_id: str = Field(..., pattern="^[A-Z0-9_]+$", description="Powder batch identifier")
    powder_condition: str = Field(..., description="Condition of powder")
    powder_particle_size: Optional[float] = Field(None, ge=0, le=200, description="Particle size in μm")
    powder_flowability: Optional[float] = Field(None, ge=0, le=100, description="Powder flowability score")


class QualityMetrics(BaseModel):
    """Quality metrics for manufactured parts."""
    density: float = Field(..., ge=0, le=100, description="Density percentage")
    surface_roughness: float = Field(..., ge=0, le=100, description="Surface roughness in μm")
    dimensional_accuracy: float = Field(..., ge=0, le=100, description="Dimensional accuracy percentage")
    defect_count: int = Field(..., ge=0, description="Number of defects")
    quality_score: float = Field(..., ge=0, le=100, description="Overall quality score")
    quality_status: QualityStatus = Field(..., description="Quality status")


class SensorLocation(BaseModel):
    """Sensor location coordinates."""
    x_coordinate: float = Field(..., description="X coordinate in mm")
    y_coordinate: float = Field(..., description="Y coordinate in mm")
    z_coordinate: float = Field(..., description="Z coordinate in mm")
    zone: str = Field(..., description="Zone identifier")


class MeasurementData(BaseModel):
    """Sensor measurement data."""
    value: float = Field(..., description="Measured value")
    unit: str = Field(..., description="Measurement unit")
    precision: float = Field(..., ge=0, le=100, description="Measurement precision")
    accuracy: float = Field(..., ge=0, le=100, description="Measurement accuracy")
    calibration_factor: Optional[float] = Field(None, description="Calibration factor")


class EnvironmentalConditions(BaseModel):
    """Environmental conditions during measurement."""
    temperature: float = Field(..., description="Temperature in °C")
    humidity: float = Field(..., ge=0, le=100, description="Humidity percentage")
    pressure: float = Field(..., ge=0, description="Pressure in Pa")
    vibration: Optional[float] = Field(None, ge=0, description="Vibration level")


class QualityFlags(BaseModel):
    """Quality flags for sensor data."""
    is_valid: bool = Field(..., description="Data validity flag")
    is_calibrated: bool = Field(..., description="Calibration status")
    is_outlier: bool = Field(..., description="Outlier detection flag")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence score")


class ProcessContext(BaseModel):
    """Process context for sensor data."""
    process_id: str = Field(..., pattern="^[A-Z0-9_]+$", description="Process identifier")
    build_id: Optional[str] = Field(None, pattern="^[A-Z0-9_]+$", description="Build identifier")
    layer_number: Optional[int] = Field(None, ge=0, description="Layer number")
    scan_line: Optional[int] = Field(None, ge=0, description="Scan line number")


class DimensionalMetrics(BaseModel):
    """Dimensional quality metrics."""
    length: float = Field(..., ge=0, description="Length in mm")
    width: float = Field(..., ge=0, description="Width in mm")
    height: float = Field(..., ge=0, description="Height in mm")
    tolerance: float = Field(..., ge=0, description="Tolerance in mm")
    deviation: float = Field(..., description="Deviation from target in mm")


class SurfaceQuality(BaseModel):
    """Surface quality metrics."""
    roughness_ra: float = Field(..., ge=0, description="Ra roughness in μm")
    roughness_rz: float = Field(..., ge=0, description="Rz roughness in μm")
    waviness: float = Field(..., ge=0, description="Waviness in μm")
    porosity: float = Field(..., ge=0, le=100, description="Porosity percentage")
    density: float = Field(..., ge=0, le=100, description="Density percentage")


class MechanicalProperties(BaseModel):
    """Mechanical properties of the material."""
    tensile_strength: float = Field(..., ge=0, description="Tensile strength in MPa")
    yield_strength: float = Field(..., ge=0, description="Yield strength in MPa")
    hardness: float = Field(..., ge=0, description="Hardness in HV")
    elastic_modulus: float = Field(..., ge=0, description="Elastic modulus in GPa")
    fatigue_limit: Optional[float] = Field(None, ge=0, description="Fatigue limit in MPa")


class DefectAnalysis(BaseModel):
    """Defect analysis results."""
    defect_count: int = Field(..., ge=0, description="Number of defects")
    defect_density: float = Field(..., ge=0, description="Defect density per cm³")
    defect_types: List[str] = Field(..., description="Types of defects")
    critical_defects: int = Field(..., ge=0, description="Number of critical defects")
    defect_severity: str = Field(..., description="Overall defect severity")


class QualityScores(BaseModel):
    """Quality scoring system."""
    overall_score: float = Field(..., ge=0, le=100, description="Overall quality score")
    dimensional_score: float = Field(..., ge=0, le=100, description="Dimensional quality score")
    surface_score: float = Field(..., ge=0, le=100, description="Surface quality score")
    mechanical_score: float = Field(..., ge=0, le=100, description="Mechanical quality score")
    defect_score: float = Field(..., ge=0, le=100, description="Defect quality score")


class MeasurementConditions(BaseModel):
    """Measurement conditions and equipment."""
    temperature: float = Field(..., description="Measurement temperature in °C")
    humidity: float = Field(..., ge=0, le=100, description="Measurement humidity percentage")
    measurement_equipment: str = Field(..., description="Equipment used for measurement")
    operator_id: str = Field(..., pattern="^[A-Z0-9_]+$", description="Operator identifier")
    measurement_standard: str = Field(..., description="Measurement standard used")


class MaterialProperties(BaseModel):
    """Material properties and batch information."""
    material_type: MaterialType = Field(..., description="Type of material")
    powder_batch: str = Field(..., pattern="^[A-Z0-9_]+$", description="Powder batch identifier")
    heat_treatment: Optional[str] = Field(None, description="Heat treatment applied")
    post_processing: Optional[str] = Field(None, description="Post-processing operations")


class LaserParameters(BaseModel):
    """Laser system parameters."""
    laser_power: float = Field(..., ge=0, le=1000, description="Laser power in watts")
    scan_speed: float = Field(..., ge=0, le=10000, description="Scan speed in mm/s")
    hatch_spacing: float = Field(..., ge=0.01, le=1.0, description="Hatch spacing in mm")
    exposure_time: float = Field(..., ge=0, le=1000, description="Exposure time in ms")


class GeometryData(BaseModel):
    """Geometry data for build instructions."""
    contour_paths: str = Field(..., description="Contour path data")
    hatch_patterns: str = Field(..., description="Hatch pattern data")
    support_structures: Optional[str] = Field(None, description="Support structure data")


class PowderRequirements(BaseModel):
    """Powder requirements for build."""
    powder_type: str = Field(..., description="Type of powder")
    powder_amount: float = Field(..., ge=0, description="Amount of powder in kg")
    powder_condition: str = Field(..., description="Condition of powder")


class BuildPlateRequirements(BaseModel):
    """Build plate requirements."""
    plate_material: str = Field(..., description="Build plate material")
    plate_temperature: float = Field(..., ge=0, le=500, description="Plate temperature in °C")
    plate_preparation: str = Field(..., description="Plate preparation steps")


class HeatTreatment(BaseModel):
    """Heat treatment parameters."""
    temperature: float = Field(..., ge=0, le=2000, description="Temperature in °C")
    duration: float = Field(..., ge=0, description="Duration in hours")
    atmosphere: str = Field(..., description="Atmosphere type")


class PerformanceMetrics(BaseModel):
    """Performance metrics for analytics."""
    throughput: float = Field(..., ge=0, description="Throughput in parts/hour")
    efficiency: float = Field(..., ge=0, le=100, description="Efficiency percentage")
    utilization: float = Field(..., ge=0, le=100, description="Utilization percentage")
    downtime: int = Field(..., ge=0, description="Downtime in minutes")
    uptime: int = Field(..., ge=0, description="Uptime in minutes")


class QualityAnalytics(BaseModel):
    """Quality analytics metrics."""
    defect_rate: float = Field(..., ge=0, le=100, description="Defect rate percentage")
    quality_score: float = Field(..., ge=0, le=100, description="Quality score")
    rework_rate: float = Field(..., ge=0, le=100, description="Rework rate percentage")
    scrap_rate: float = Field(..., ge=0, le=100, description="Scrap rate percentage")
    first_pass_yield: float = Field(..., ge=0, le=100, description="First pass yield percentage")


class CostAnalytics(BaseModel):
    """Cost analytics metrics."""
    material_cost: float = Field(..., ge=0, description="Material cost")
    energy_cost: float = Field(..., ge=0, description="Energy cost")
    labor_cost: float = Field(..., ge=0, description="Labor cost")
    maintenance_cost: float = Field(..., ge=0, description="Maintenance cost")
    total_cost: float = Field(..., ge=0, description="Total cost")


class TrendAnalysis(BaseModel):
    """Trend analysis results."""
    trend_direction: str = Field(..., description="Trend direction")
    trend_magnitude: float = Field(..., description="Trend magnitude")
    trend_confidence: float = Field(..., ge=0, le=1, description="Trend confidence")
    has_seasonality: bool = Field(..., description="Has seasonality")
    seasonal_period: Optional[str] = Field(None, description="Seasonal period")
    seasonal_strength: Optional[float] = Field(None, ge=0, le=1, description="Seasonal strength")


class PredictiveAnalytics(BaseModel):
    """Predictive analytics results."""
    prediction_type: str = Field(..., description="Type of prediction")
    predicted_value: float = Field(..., description="Predicted value")
    lower_bound: float = Field(..., description="Lower confidence bound")
    upper_bound: float = Field(..., description="Upper confidence bound")
    prediction_horizon: int = Field(..., ge=0, description="Prediction horizon in hours")
    model_accuracy: float = Field(..., ge=0, le=1, description="Model accuracy")


class AnomalyDetection(BaseModel):
    """Anomaly detection results."""
    anomaly_score: float = Field(..., ge=0, le=1, description="Anomaly score")
    anomaly_type: str = Field(..., description="Type of anomaly")
    severity: AlertSeverity = Field(..., description="Anomaly severity")
    detection_method: str = Field(..., description="Detection method used")
    affected_metrics: List[str] = Field(..., description="Affected metrics")


class KPIMetrics(BaseModel):
    """KPI metrics for analytics."""
    kpi_name: str = Field(..., description="KPI name")
    kpi_value: float = Field(..., description="KPI value")
    kpi_target: float = Field(..., description="KPI target")
    kpi_variance: float = Field(..., description="KPI variance")
    kpi_status: str = Field(..., description="KPI status")


class SearchContext(BaseModel):
    """Search context for search logs."""
    search_type: str = Field(..., description="Type of search")
    search_intent: str = Field(..., description="Search intent")
    date_range: Optional[Dict[str, datetime]] = Field(None, description="Date range filter")
    material_type: Optional[str] = Field(None, description="Material type filter")
    process_id: Optional[str] = Field(None, description="Process ID filter")
    quality_range: Optional[Dict[str, float]] = Field(None, description="Quality range filter")


class SearchResults(BaseModel):
    """Search results metadata."""
    total_hits: int = Field(..., ge=0, description="Total number of hits")
    returned_hits: int = Field(..., ge=0, description="Number of returned hits")
    search_time: int = Field(..., ge=0, description="Search time in ms")
    result_relevance: float = Field(..., ge=0, le=1, description="Result relevance score")
    result_categories: List[str] = Field(..., description="Result categories")


class UserInteraction(BaseModel):
    """User interaction data."""
    clicked_results: int = Field(..., ge=0, description="Number of clicked results")
    clicked_positions: int = Field(..., ge=0, description="Average clicked position")
    refinement_queries: int = Field(..., ge=0, description="Number of refinement queries")
    session_duration: int = Field(..., ge=0, description="Session duration in seconds")
    satisfaction_score: Optional[float] = Field(None, ge=0, le=1, description="Satisfaction score")


class SearchPerformanceMetrics(BaseModel):
    """Performance metrics for search logs."""
    query_complexity: str = Field(..., description="Query complexity level")
    index_used: str = Field(..., description="Index used for search")
    cache_hit: bool = Field(..., description="Cache hit status")
    aggregation_time: Optional[int] = Field(None, ge=0, description="Aggregation time in ms")
    sorting_time: Optional[int] = Field(None, ge=0, description="Sorting time in ms")


class ErrorHandling(BaseModel):
    """Error handling information."""
    has_errors: bool = Field(..., description="Has errors flag")
    error_type: Optional[str] = Field(None, description="Error type")
    error_message: Optional[str] = Field(None, description="Error message")
    error_severity: Optional[AlertSeverity] = Field(None, description="Error severity")
    retry_count: int = Field(default=0, ge=0, description="Number of retries")


class TechnicalDetails(BaseModel):
    """Technical details for search logs."""
    client_ip: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="User agent string")
    request_size: int = Field(..., ge=0, description="Request size in bytes")
    response_size: int = Field(..., ge=0, description="Response size in bytes")
    api_version: str = Field(..., description="API version")
    request_id: str = Field(..., description="Request identifier")


# PBF Process Document Model
class PBFProcessDocument(BaseElasticsearchDocument):
    """PBF process document model matching pbf_process_index.json."""
    
    document_type: Literal["pbf_process"] = "pbf_process"
    timestamp: datetime = Field(..., description="Process timestamp")
    process_parameters: ProcessParameters = Field(..., description="Process parameters")
    material_info: MaterialInfo = Field(..., description="Material information")
    quality_metrics: QualityMetrics = Field(..., description="Quality metrics")
    full_text_search: Optional[str] = Field(None, description="Full text search content")
    
    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v):
        """Validate timestamp is not in the future."""
        if v > datetime.utcnow():
            raise ValueError("Timestamp cannot be in the future")
        return v


# Sensor Readings Document Model
class SensorReadingsDocument(BaseElasticsearchDocument):
    """Sensor readings document model matching sensor_readings_index.json."""
    
    document_type: Literal["sensor_readings"] = "sensor_readings"
    sensor_id: str = Field(..., pattern="^[A-Z0-9_]+$", description="Sensor identifier")
    timestamp: datetime = Field(..., description="Sensor reading timestamp")
    sensor_type: SensorType = Field(..., description="Type of sensor")
    sensor_location: SensorLocation = Field(..., description="Sensor location")
    measurement_data: MeasurementData = Field(..., description="Measurement data")
    environmental_conditions: EnvironmentalConditions = Field(..., description="Environmental conditions")
    quality_flags: QualityFlags = Field(..., description="Quality flags")
    process_context: ProcessContext = Field(..., description="Process context")


# Quality Metrics Document Model
class QualityMetricsDocument(BaseElasticsearchDocument):
    """Quality metrics document model matching quality_metrics_index.json."""
    
    document_type: Literal["quality_metrics"] = "quality_metrics"
    quality_id: str = Field(..., pattern="^[A-Z0-9_]+$", description="Quality identifier")
    timestamp: datetime = Field(..., description="Quality measurement timestamp")
    measurement_type: str = Field(..., description="Type of measurement")
    dimensional_metrics: DimensionalMetrics = Field(..., description="Dimensional metrics")
    surface_quality: SurfaceQuality = Field(..., description="Surface quality metrics")
    mechanical_properties: MechanicalProperties = Field(..., description="Mechanical properties")
    defect_analysis: DefectAnalysis = Field(..., description="Defect analysis")
    quality_scores: QualityScores = Field(..., description="Quality scores")
    measurement_conditions: MeasurementConditions = Field(..., description="Measurement conditions")
    material_properties: MaterialProperties = Field(..., description="Material properties")


# Machine Status Document Model
class MachineStatusDocument(BaseElasticsearchDocument):
    """Machine status document model matching machine_status_index.json."""
    
    document_type: Literal["machine_status"] = "machine_status"
    machine_id: str = Field(..., pattern="^[A-Z0-9_]+$", description="Machine identifier")
    timestamp: datetime = Field(..., description="Status timestamp")
    status: MachineStatus = Field(..., description="Machine status")
    operational_state: Dict[str, Any] = Field(..., description="Operational state")
    system_health: Dict[str, Any] = Field(..., description="System health metrics")
    laser_system: Dict[str, Any] = Field(..., description="Laser system status")
    build_platform: Dict[str, Any] = Field(..., description="Build platform status")
    powder_system: Dict[str, Any] = Field(..., description="Powder system status")
    environmental_conditions: Dict[str, Any] = Field(..., description="Environmental conditions")
    alerts_and_warnings: Dict[str, Any] = Field(..., description="Alerts and warnings")
    performance_metrics: PerformanceMetrics = Field(..., description="Performance metrics")
    maintenance_info: Dict[str, Any] = Field(..., description="Maintenance information")


# Build Instructions Document Model
class BuildInstructionsDocument(BaseElasticsearchDocument):
    """Build instructions document model matching build_instructions_index.json."""
    
    document_type: Literal["build_instructions"] = "build_instructions"
    instruction_id: str = Field(..., pattern="^[A-Z0-9_]+$", description="Instruction identifier")
    timestamp: datetime = Field(..., description="Instruction timestamp")
    instruction_type: str = Field(..., description="Type of instruction")
    instruction_content: str = Field(..., description="Instruction content")
    layer_instructions: Dict[str, Any] = Field(..., description="Layer instructions")
    material_requirements: Dict[str, Any] = Field(..., description="Material requirements")
    quality_requirements: Dict[str, Any] = Field(..., description="Quality requirements")
    process_parameters: Dict[str, Any] = Field(..., description="Process parameters")
    support_structures: Dict[str, Any] = Field(..., description="Support structures")
    post_processing: Dict[str, Any] = Field(..., description="Post-processing requirements")
    safety_requirements: Dict[str, Any] = Field(..., description="Safety requirements")


# Analytics Document Model
class AnalyticsDocument(BaseElasticsearchDocument):
    """Analytics document model matching analytics_index.json."""
    
    document_type: Literal["analytics"] = "analytics"
    analytics_id: str = Field(..., pattern="^[A-Z0-9_]+$", description="Analytics identifier")
    timestamp: datetime = Field(..., description="Analytics timestamp")
    analysis_type: str = Field(..., description="Type of analysis")
    data_source: str = Field(..., description="Data source")
    performance_metrics: PerformanceMetrics = Field(..., description="Performance metrics")
    quality_analytics: QualityAnalytics = Field(..., description="Quality analytics")
    cost_analytics: CostAnalytics = Field(..., description="Cost analytics")
    trend_analysis: TrendAnalysis = Field(..., description="Trend analysis")
    predictive_analytics: PredictiveAnalytics = Field(..., description="Predictive analytics")
    anomaly_detection: AnomalyDetection = Field(..., description="Anomaly detection")
    comparative_analysis: Dict[str, Any] = Field(..., description="Comparative analysis")
    kpi_metrics: KPIMetrics = Field(..., description="KPI metrics")
    reporting_metadata: Dict[str, Any] = Field(..., description="Reporting metadata")


# Search Logs Document Model
class SearchLogsDocument(BaseElasticsearchDocument):
    """Search logs document model matching search_logs_index.json."""
    
    document_type: Literal["search_logs"] = "search_logs"
    log_id: str = Field(..., pattern="^[A-Z0-9_]+$", description="Log identifier")
    timestamp: datetime = Field(..., description="Search timestamp")
    user_id: str = Field(..., pattern="^[A-Z0-9_]+$", description="User identifier")
    session_id: str = Field(..., pattern="^[A-Z0-9_]+$", description="Session identifier")
    search_query: str = Field(..., description="Search query")
    search_context: SearchContext = Field(..., description="Search context")
    search_results: SearchResults = Field(..., description="Search results")
    user_interaction: UserInteraction = Field(..., description="User interaction")
    performance_metrics: SearchPerformanceMetrics = Field(..., description="Performance metrics")
    error_handling: ErrorHandling = Field(..., description="Error handling")
    search_analytics: Dict[str, Any] = Field(..., description="Search analytics")
    technical_details: TechnicalDetails = Field(..., description="Technical details")


# Document Factory
class ElasticsearchDocumentFactory:
    """Factory for creating Elasticsearch document models."""
    
    @staticmethod
    def create_document(document_type: str, data: Dict[str, Any]) -> BaseElasticsearchDocument:
        """Create a document model based on document type."""
        document_classes = {
            "pbf_process": PBFProcessDocument,
            "sensor_readings": SensorReadingsDocument,
            "quality_metrics": QualityMetricsDocument,
            "machine_status": MachineStatusDocument,
            "build_instructions": BuildInstructionsDocument,
            "analytics": AnalyticsDocument,
            "search_logs": SearchLogsDocument,
        }
        
        if document_type not in document_classes:
            raise ValueError(f"Unknown document type: {document_type}")
        
        return document_classes[document_type](**data)
    
    @staticmethod
    def validate_document(document: BaseElasticsearchDocument) -> bool:
        """Validate a document model."""
        try:
            document.model_validate(document.model_dump())
            return True
        except Exception:
            return False


# Export all models
__all__ = [
    "BaseElasticsearchDocument",
    "PBFProcessDocument",
    "SensorReadingsDocument", 
    "QualityMetricsDocument",
    "MachineStatusDocument",
    "BuildInstructionsDocument",
    "AnalyticsDocument",
    "SearchLogsDocument",
    "ElasticsearchDocumentFactory",
    "ProcessStatus",
    "MachineStatus",
    "SensorType",
    "MaterialType",
    "QualityStatus",
    "AlertSeverity",
    "ProcessParameters",
    "MaterialInfo",
    "QualityMetrics",
    "SensorLocation",
    "MeasurementData",
    "EnvironmentalConditions",
    "QualityFlags",
    "ProcessContext",
    "DimensionalMetrics",
    "SurfaceQuality",
    "MechanicalProperties",
    "DefectAnalysis",
    "QualityScores",
    "MeasurementConditions",
    "MaterialProperties",
    "LaserParameters",
    "GeometryData",
    "PowderRequirements",
    "BuildPlateRequirements",
    "HeatTreatment",
    "PerformanceMetrics",
    "QualityAnalytics",
    "CostAnalytics",
    "TrendAnalysis",
    "PredictiveAnalytics",
    "AnomalyDetection",
    "KPIMetrics",
    "SearchContext",
    "SearchResults",
    "UserInteraction",
    "ErrorHandling",
    "TechnicalDetails"
]
