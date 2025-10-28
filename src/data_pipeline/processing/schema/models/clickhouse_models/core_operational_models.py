"""
ClickHouse Core Operational Models

This module contains Pydantic models for core operational data in ClickHouse data warehouse.
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator


# ============================================================================
# ENUMS
# ============================================================================

class ProcessStatus(str, Enum):
    """Process status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class AtmosphereType(str, Enum):
    """Atmosphere type enumeration."""
    ARGON = "argon"
    NITROGEN = "nitrogen"
    HELIUM = "helium"
    VACUUM = "vacuum"
    AIR = "air"

class QualityGrade(str, Enum):
    """Quality grade enumeration."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

class MachineStatusType(str, Enum):
    """Machine status type enumeration."""
    OPERATIONAL = "operational"
    MAINTENANCE = "maintenance"
    IDLE = "idle"
    ERROR = "error"
    OFFLINE = "offline"

class HealthStatus(str, Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class LaserStatus(str, Enum):
    """Laser status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    CALIBRATING = "calibrating"
    ERROR = "error"

class PlatformStatus(str, Enum):
    """Platform status enumeration."""
    READY = "ready"
    MOVING = "moving"
    POSITIONED = "positioned"
    ERROR = "error"

class PowderStatus(str, Enum):
    """Powder status enumeration."""
    AVAILABLE = "available"
    LOW = "low"
    EMPTY = "empty"
    CONTAMINATED = "contaminated"

class AlertSeverity(str, Enum):
    """Alert severity enumeration."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MaintenanceType(str, Enum):
    """Maintenance type enumeration."""
    PREVENTIVE = "preventive"
    CORRECTIVE = "corrective"
    PREDICTIVE = "predictive"
    EMERGENCY = "emergency"

class SensorType(str, Enum):
    """Sensor type enumeration."""
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    VIBRATION = "vibration"
    LASER_POWER = "laser_power"
    POSITION = "position"
    FLOW = "flow"
    HUMIDITY = "humidity"
    OXYGEN = "oxygen"

class UnitType(str, Enum):
    """Unit type enumeration."""
    CELSIUS = "celsius"
    KELVIN = "kelvin"
    FAHRENHEIT = "fahrenheit"
    PASCAL = "pascal"
    BAR = "bar"
    PSI = "psi"
    HERTZ = "hertz"
    WATT = "watt"
    MILLIMETER = "millimeter"
    PERCENT = "percent"
    PPM = "ppm"

class AnalysisType(str, Enum):
    """Analysis type enumeration."""
    PERFORMANCE = "performance"
    QUALITY = "quality"
    COST = "cost"
    TREND = "trend"
    PREDICTIVE = "predictive"
    ANOMALY = "anomaly"
    COMPARATIVE = "comparative"

class TrendDirection(str, Enum):
    """Trend direction enumeration."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"

class AnomalyType(str, Enum):
    """Anomaly type enumeration."""
    SPIKES = "spikes"
    DROPS = "drops"
    PATTERN_BREAK = "pattern_break"
    OUTLIER = "outlier"
    TREND_CHANGE = "trend_change"

class ComparisonType(str, Enum):
    """Comparison type enumeration."""
    HISTORICAL = "historical"
    BASELINE = "baseline"
    PEER = "peer"
    TARGET = "target"


# ============================================================================
# COMPONENT MODELS
# ============================================================================

class ProcessParameters(BaseModel):
    """Process parameters model."""
    layer_number: int = Field(..., ge=0, le=10000, description="Layer number")
    temperature: float = Field(..., ge=0, le=2000, description="Temperature in Celsius")
    pressure: float = Field(..., ge=0, le=1000, description="Pressure in bar")
    laser_power: float = Field(..., ge=0, le=1000, description="Laser power in watts")
    scan_speed: float = Field(..., ge=0, le=10000, description="Scan speed in mm/s")
    layer_height: float = Field(..., ge=0.01, le=1.0, description="Layer height in mm")
    hatch_spacing: Optional[float] = Field(None, ge=0.01, le=1.0, description="Hatch spacing in mm")
    exposure_time: Optional[float] = Field(None, ge=0, le=3600, description="Exposure time in seconds")

class MaterialInfo(BaseModel):
    """Material information model."""
    material_type: str = Field(..., description="Material type")
    powder_batch_id: Optional[str] = Field(None, description="Powder batch ID")
    powder_condition: Optional[str] = Field(None, description="Powder condition")
    material_properties: Optional[Dict[str, Any]] = Field(None, description="Material properties")

class QualityMetrics(BaseModel):
    """Quality metrics model."""
    dimensional_accuracy: float = Field(..., ge=0, le=100, description="Dimensional accuracy percentage")
    surface_roughness: float = Field(..., ge=0, description="Surface roughness in micrometers")
    density: float = Field(..., ge=0, le=100, description="Density percentage")
    tensile_strength: Optional[float] = Field(None, ge=0, description="Tensile strength in MPa")
    yield_strength: Optional[float] = Field(None, ge=0, description="Yield strength in MPa")
    hardness: Optional[float] = Field(None, ge=0, description="Hardness in HV")

class OperationalState(BaseModel):
    """Operational state model."""
    status: MachineStatusType = Field(..., description="Machine status")
    uptime: float = Field(..., ge=0, description="Uptime in hours")
    last_maintenance: Optional[datetime] = Field(None, description="Last maintenance timestamp")
    next_maintenance: Optional[datetime] = Field(None, description="Next maintenance timestamp")

class SystemHealth(BaseModel):
    """System health model."""
    overall_health: HealthStatus = Field(..., description="Overall system health")
    cpu_usage: float = Field(..., ge=0, le=100, description="CPU usage percentage")
    memory_usage: float = Field(..., ge=0, le=100, description="Memory usage percentage")
    disk_usage: float = Field(..., ge=0, le=100, description="Disk usage percentage")
    network_status: str = Field(..., description="Network status")

class LaserSystem(BaseModel):
    """Laser system model."""
    status: LaserStatus = Field(..., description="Laser status")
    power_level: float = Field(..., ge=0, le=100, description="Power level percentage")
    wavelength: float = Field(..., ge=0, description="Wavelength in nm")
    beam_quality: Optional[float] = Field(None, ge=0, le=1, description="Beam quality factor")
    last_calibration: Optional[datetime] = Field(None, description="Last calibration timestamp")

class BuildPlatform(BaseModel):
    """Build platform model."""
    status: PlatformStatus = Field(..., description="Platform status")
    position_x: float = Field(..., description="X position in mm")
    position_y: float = Field(..., description="Y position in mm")
    position_z: float = Field(..., description="Z position in mm")
    temperature: Optional[float] = Field(None, description="Platform temperature in Celsius")

class PowderSystem(BaseModel):
    """Powder system model."""
    status: PowderStatus = Field(..., description="Powder status")
    level: float = Field(..., ge=0, le=100, description="Powder level percentage")
    flow_rate: Optional[float] = Field(None, ge=0, description="Flow rate in g/min")
    particle_size: Optional[float] = Field(None, ge=0, description="Particle size in micrometers")

class EnvironmentalConditions(BaseModel):
    """Environmental conditions model."""
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Humidity percentage")
    pressure: float = Field(..., ge=0, description="Pressure in bar")
    oxygen_level: Optional[float] = Field(None, ge=0, le=100, description="Oxygen level percentage")

class AlertInfo(BaseModel):
    """Alert information model."""
    alert_id: str = Field(..., description="Alert ID")
    severity: AlertSeverity = Field(..., description="Alert severity")
    message: str = Field(..., description="Alert message")
    timestamp: datetime = Field(..., description="Alert timestamp")
    resolved: bool = Field(False, description="Alert resolved status")

class PerformanceMetrics(BaseModel):
    """Performance metrics model."""
    throughput: float = Field(..., ge=0, description="Throughput in parts/hour")
    efficiency: float = Field(..., ge=0, le=100, description="Efficiency percentage")
    energy_consumption: float = Field(..., ge=0, description="Energy consumption in kWh")
    cycle_time: float = Field(..., ge=0, description="Cycle time in minutes")

class MaintenanceInfo(BaseModel):
    """Maintenance information model."""
    maintenance_type: MaintenanceType = Field(..., description="Maintenance type")
    scheduled_date: Optional[datetime] = Field(None, description="Scheduled maintenance date")
    duration: Optional[float] = Field(None, ge=0, description="Maintenance duration in hours")
    cost: Optional[float] = Field(None, ge=0, description="Maintenance cost")
    technician: Optional[str] = Field(None, description="Technician name")

class QualityScore(BaseModel):
    """Quality score model."""
    overall_score: float = Field(..., ge=0, le=100, description="Overall quality score")
    dimensional_score: float = Field(..., ge=0, le=100, description="Dimensional quality score")
    surface_score: float = Field(..., ge=0, le=100, description="Surface quality score")
    material_score: float = Field(..., ge=0, le=100, description="Material quality score")

class PerformanceAnalysis(BaseModel):
    """Performance analysis model."""
    metric_name: str = Field(..., description="Performance metric name")
    current_value: float = Field(..., description="Current metric value")
    target_value: float = Field(..., description="Target metric value")
    variance: float = Field(..., description="Variance from target")
    trend: TrendDirection = Field(..., description="Performance trend")

class QualityAnalysis(BaseModel):
    """Quality analysis model."""
    quality_grade: QualityGrade = Field(..., description="Quality grade")
    defect_count: int = Field(..., ge=0, description="Defect count")
    defect_rate: float = Field(..., ge=0, le=100, description="Defect rate percentage")
    improvement_areas: List[str] = Field(..., description="Areas for improvement")

class CostAnalysis(BaseModel):
    """Cost analysis model."""
    material_cost: float = Field(..., ge=0, description="Material cost")
    energy_cost: float = Field(..., ge=0, description="Energy cost")
    labor_cost: float = Field(..., ge=0, description="Labor cost")
    total_cost: float = Field(..., ge=0, description="Total cost")
    cost_per_part: float = Field(..., ge=0, description="Cost per part")

class TrendAnalysis(BaseModel):
    """Trend analysis model."""
    metric_name: str = Field(..., description="Trend metric name")
    direction: TrendDirection = Field(..., description="Trend direction")
    magnitude: float = Field(..., description="Trend magnitude")
    confidence: float = Field(..., ge=0, le=1, description="Trend confidence")

class PredictiveAnalysis(BaseModel):
    """Predictive analysis model."""
    prediction_type: str = Field(..., description="Prediction type")
    predicted_value: float = Field(..., description="Predicted value")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    time_horizon: float = Field(..., ge=0, description="Time horizon in hours")

class AnomalyDetection(BaseModel):
    """Anomaly detection model."""
    anomaly_type: AnomalyType = Field(..., description="Anomaly type")
    severity: float = Field(..., ge=0, le=1, description="Anomaly severity")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    description: str = Field(..., description="Anomaly description")

class ComparativeAnalysis(BaseModel):
    """Comparative analysis model."""
    comparison_type: ComparisonType = Field(..., description="Comparison type")
    baseline_value: float = Field(..., description="Baseline value")
    current_value: float = Field(..., description="Current value")
    difference: float = Field(..., description="Difference from baseline")
    percentage_change: float = Field(..., description="Percentage change")


# ============================================================================
# MAIN MODELS
# ============================================================================

class PBFProcessModel(BaseModel):
    """PBF Process model for ClickHouse - matches pbf_processes.sql schema with optional fields."""
    # Primary identifiers
    id: int = Field(..., description="Primary key ID")
    process_id: str = Field(..., description="Process ID")
    build_id: Optional[str] = Field(None, description="Build ID")
    part_id: Optional[str] = Field(None, description="Part ID")
    
    # Timestamps
    timestamp: datetime = Field(..., description="Process timestamp")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Update timestamp")
    
    # Process parameters (flat fields to match SQL schema)
    laser_power: Optional[float] = Field(None, description="Laser power in watts")
    scan_speed: Optional[float] = Field(None, description="Scan speed in mm/s")
    layer_thickness: Optional[float] = Field(None, description="Layer thickness in mm")
    hatch_spacing: Optional[float] = Field(None, description="Hatch spacing in mm")
    build_plate_temp: Optional[float] = Field(None, description="Build plate temperature")
    exposure_time: Optional[float] = Field(None, description="Exposure time in seconds")
    focus_offset: Optional[float] = Field(None, description="Focus offset")
    
    # Material information (flat fields to match SQL schema)
    material_type: Optional[str] = Field(None, description="Material type")
    powder_batch_id: Optional[str] = Field(None, description="Powder batch ID")
    powder_condition: Optional[str] = Field(None, description="Powder condition")
    powder_particle_size: Optional[float] = Field(None, description="Powder particle size")
    powder_flowability: Optional[float] = Field(None, description="Powder flowability")
    
    # Quality metrics (flat fields to match SQL schema)
    density: Optional[float] = Field(None, description="Density")
    surface_roughness: Optional[float] = Field(None, description="Surface roughness")
    dimensional_accuracy: Optional[float] = Field(None, description="Dimensional accuracy")
    defect_count: Optional[int] = Field(None, description="Defect count")
    quality_score: Optional[float] = Field(None, description="Quality score")
    quality_status: Optional[str] = Field(None, description="Quality status")
    
    # Operational metadata
    operator_id: Optional[str] = Field(None, description="Operator ID")
    machine_id: Optional[str] = Field(None, description="Machine ID")
    build_job_id: Optional[str] = Field(None, description="Build job ID")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class MachineStatusModel(BaseModel):
    """Machine Status model for ClickHouse - matches machine_status.sql schema with optional fields."""
    # Primary identifiers
    id: int = Field(..., description="Primary key ID")
    machine_id: str = Field(..., description="Machine ID")
    timestamp: datetime = Field(..., description="Status timestamp")
    
    # Operational state (flattened)
    status: Optional[str] = Field(None, description="Machine status")
    current_state: Optional[str] = Field(None, description="Current state")
    previous_state: Optional[str] = Field(None, description="Previous state")
    state_duration: Optional[int] = Field(None, description="State duration in seconds")
    state_transitions: Optional[int] = Field(None, description="Number of state transitions")
    
    # System health metrics (flattened)
    overall_health: Optional[float] = Field(None, description="Overall health score")
    cpu_usage: Optional[float] = Field(None, description="CPU usage percentage")
    memory_usage: Optional[float] = Field(None, description="Memory usage percentage")
    disk_usage: Optional[float] = Field(None, description="Disk usage percentage")
    network_status: Optional[str] = Field(None, description="Network status")
    
    # Laser system metrics (flattened)
    laser_power: Optional[float] = Field(None, description="Laser power in watts")
    laser_temperature: Optional[float] = Field(None, description="Laser temperature in Celsius")
    laser_status: Optional[str] = Field(None, description="Laser status")
    laser_hours: Optional[int] = Field(None, description="Laser operating hours")
    laser_wavelength: Optional[float] = Field(None, description="Laser wavelength in nm")
    
    # Build platform metrics (flattened)
    platform_temperature: Optional[float] = Field(None, description="Platform temperature in Celsius")
    platform_x: Optional[float] = Field(None, description="Platform X position in mm")
    platform_y: Optional[float] = Field(None, description="Platform Y position in mm")
    platform_z: Optional[float] = Field(None, description="Platform Z position in mm")
    platform_status: Optional[str] = Field(None, description="Platform status")
    
    # Powder system metrics (flattened)
    powder_level: Optional[float] = Field(None, description="Powder level percentage")
    powder_temperature: Optional[float] = Field(None, description="Powder temperature in Celsius")
    powder_flow_rate: Optional[float] = Field(None, description="Powder flow rate in g/min")
    powder_status: Optional[str] = Field(None, description="Powder status")
    powder_quality: Optional[str] = Field(None, description="Powder quality")
    
    # Environmental conditions (flattened)
    chamber_temperature: Optional[float] = Field(None, description="Chamber temperature in Celsius")
    chamber_humidity: Optional[float] = Field(None, description="Chamber humidity percentage")
    oxygen_level: Optional[float] = Field(None, description="Oxygen level percentage")
    pressure: Optional[float] = Field(None, description="Pressure in bar")
    
    # Alerts and warnings (flattened)
    active_alerts: Optional[int] = Field(None, description="Number of active alerts")
    alert_level: Optional[str] = Field(None, description="Alert level")
    alert_types: Optional[List[str]] = Field(None, description="Alert types")
    last_maintenance: Optional[datetime] = Field(None, description="Last maintenance timestamp")
    next_maintenance: Optional[datetime] = Field(None, description="Next maintenance timestamp")
    
    # Performance metrics (flattened)
    throughput: Optional[float] = Field(None, description="Throughput in parts/hour")
    efficiency: Optional[float] = Field(None, description="Efficiency percentage")
    utilization: Optional[float] = Field(None, description="Utilization percentage")
    downtime: Optional[int] = Field(None, description="Downtime in seconds")
    uptime: Optional[int] = Field(None, description="Uptime in seconds")
    
    # Maintenance information (flattened)
    maintenance_due: Optional[int] = Field(None, description="Maintenance due flag (0/1)")
    maintenance_type: Optional[str] = Field(None, description="Maintenance type")
    maintenance_interval: Optional[int] = Field(None, description="Maintenance interval in days")
    last_service_date: Optional[datetime] = Field(None, description="Last service date")
    service_history: Optional[List[str]] = Field(None, description="Service history")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class SensorReadingModel(BaseModel):
    """Sensor Reading model for ClickHouse - matches sensor_readings.sql schema with optional fields."""
    # Primary identifiers
    id: int = Field(..., description="Primary key ID")
    sensor_id: str = Field(..., description="Sensor ID")
    process_id: Optional[str] = Field(None, description="Process ID")
    build_id: Optional[str] = Field(None, description="Build ID")
    machine_id: Optional[str] = Field(None, description="Machine ID")
    timestamp: datetime = Field(..., description="Reading timestamp")
    
    # Sensor information
    sensor_type: Optional[str] = Field(None, description="Sensor type")
    value: Optional[float] = Field(None, description="Sensor value")
    unit: Optional[str] = Field(None, description="Measurement unit")
    location: Optional[str] = Field(None, description="Sensor location")
    status: Optional[str] = Field(None, description="Sensor status")
    quality_score: Optional[float] = Field(None, description="Quality score")
    
    # Calibration data
    calibration_date: Optional[datetime] = Field(None, description="Calibration date")
    calibration_factor: Optional[float] = Field(None, description="Calibration factor")
    calibration_accuracy: Optional[float] = Field(None, description="Calibration accuracy")
    calibration_uncertainty: Optional[float] = Field(None, description="Calibration uncertainty")
    
    # Measurement metadata
    sampling_rate: Optional[int] = Field(None, description="Sampling rate")
    data_duration: Optional[float] = Field(None, description="Data duration")
    data_points: Optional[int] = Field(None, description="Data points count")
    min_value: Optional[float] = Field(None, description="Minimum value")
    max_value: Optional[float] = Field(None, description="Maximum value")
    
    # Processing status
    processing_status: Optional[str] = Field(None, description="Processing status")
    data_quality: Optional[str] = Field(None, description="Data quality")
    file_path: Optional[str] = Field(None, description="File path")
    file_size: Optional[int] = Field(None, description="File size")
    file_hash: Optional[str] = Field(None, description="File hash")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class AnalyticsModel(BaseModel):
    """Analytics model for ClickHouse - matches analytics.sql schema with optional fields."""
    # Primary identifiers
    id: int = Field(..., description="Primary key ID")
    analytics_id: str = Field(..., description="Analytics ID")
    timestamp: datetime = Field(..., description="Analysis timestamp")
    
    # Analysis metadata (flattened)
    analysis_type: Optional[str] = Field(None, description="Analysis type")
    data_source: Optional[str] = Field(None, description="Data source")
    
    # Performance metrics (flattened)
    throughput: Optional[float] = Field(None, description="Throughput in parts/hour")
    efficiency: Optional[float] = Field(None, description="Efficiency percentage")
    utilization: Optional[float] = Field(None, description="Utilization percentage")
    downtime: Optional[int] = Field(None, description="Downtime in seconds")
    uptime: Optional[int] = Field(None, description="Uptime in seconds")
    
    # Quality analytics (flattened)
    defect_rate: Optional[float] = Field(None, description="Defect rate percentage")
    quality_score: Optional[float] = Field(None, description="Quality score")
    rework_rate: Optional[float] = Field(None, description="Rework rate percentage")
    scrap_rate: Optional[float] = Field(None, description="Scrap rate percentage")
    first_pass_yield: Optional[float] = Field(None, description="First pass yield percentage")
    
    # Cost analytics (flattened)
    material_cost: Optional[float] = Field(None, description="Material cost")
    energy_cost: Optional[float] = Field(None, description="Energy cost")
    labor_cost: Optional[float] = Field(None, description="Labor cost")
    maintenance_cost: Optional[float] = Field(None, description="Maintenance cost")
    total_cost: Optional[float] = Field(None, description="Total cost")
    
    # Trend analysis (flattened)
    trend_direction: Optional[str] = Field(None, description="Trend direction")
    trend_magnitude: Optional[float] = Field(None, description="Trend magnitude")
    trend_confidence: Optional[float] = Field(None, description="Trend confidence")
    has_seasonality: Optional[int] = Field(None, description="Has seasonality flag (0/1)")
    seasonal_period: Optional[int] = Field(None, description="Seasonal period in days")
    seasonal_strength: Optional[float] = Field(None, description="Seasonal strength")
    
    # Predictive analytics (flattened)
    prediction_type: Optional[str] = Field(None, description="Prediction type")
    predicted_value: Optional[float] = Field(None, description="Predicted value")
    lower_bound: Optional[float] = Field(None, description="Lower bound")
    upper_bound: Optional[float] = Field(None, description="Upper bound")
    prediction_horizon: Optional[int] = Field(None, description="Prediction horizon in hours")
    model_accuracy: Optional[float] = Field(None, description="Model accuracy")
    
    # Anomaly detection (flattened)
    anomaly_score: Optional[float] = Field(None, description="Anomaly score")
    anomaly_type: Optional[str] = Field(None, description="Anomaly type")
    severity: Optional[str] = Field(None, description="Severity level")
    detection_method: Optional[str] = Field(None, description="Detection method")
    affected_metrics: Optional[List[str]] = Field(None, description="Affected metrics")
    
    # Comparative analysis (flattened)
    baseline_start_date: Optional[datetime] = Field(None, description="Baseline start date")
    baseline_end_date: Optional[datetime] = Field(None, description="Baseline end date")
    comparison_start_date: Optional[datetime] = Field(None, description="Comparison start date")
    comparison_end_date: Optional[datetime] = Field(None, description="Comparison end date")
    throughput_change: Optional[float] = Field(None, description="Throughput change percentage")
    quality_change: Optional[float] = Field(None, description="Quality change percentage")
    cost_change: Optional[float] = Field(None, description="Cost change percentage")
    
    # KPI metrics (flattened)
    kpi_name: Optional[str] = Field(None, description="KPI name")
    kpi_value: Optional[float] = Field(None, description="KPI value")
    kpi_target: Optional[float] = Field(None, description="KPI target")
    kpi_status: Optional[str] = Field(None, description="KPI status")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
