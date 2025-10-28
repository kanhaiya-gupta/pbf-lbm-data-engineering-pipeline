"""
Cassandra time-series data models for PBF-LB/M data pipeline.

This module defines Pydantic models for time-series data storage in Cassandra,
optimized for high-volume sensor readings, process monitoring, and analytics.
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from pydantic import BaseModel, Field, validator, model_validator
import json
from src.data_pipeline.processing.knowledge_graph.utils.json_parser import safe_json_loads_with_fallback


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


class ProcessStatus(str, Enum):
    """Process status enumeration."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class MachineStatus(str, Enum):
    """Machine status enumeration."""
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    MAINTENANCE = "MAINTENANCE"
    ERROR = "ERROR"
    OFFLINE = "OFFLINE"


class AlertSeverity(str, Enum):
    """Alert severity enumeration."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class BaseTimeSeriesModel(BaseModel):
    """Base model for time-series data."""
    
    timestamp: datetime = Field(..., description="Timestamp of the data point")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Record creation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v)
        }


class SensorReading(BaseTimeSeriesModel):
    """Sensor reading time-series data model."""
    
    # Primary identifiers
    sensor_id: str = Field(..., description="Unique sensor identifier")
    process_id: str = Field(..., description="Process identifier")
    machine_id: str = Field(..., description="Machine identifier")
    
    # Sensor data
    sensor_type: SensorType = Field(..., description="Type of sensor")
    value: float = Field(..., description="Sensor reading value")
    unit: str = Field(..., description="Unit of measurement")
    quality_score: float = Field(..., ge=0, le=100, description="Data quality score (0-100)")
    
    # Metadata
    location: Optional[str] = Field(None, description="Sensor location")
    calibration_date: Optional[datetime] = Field(None, description="Last calibration date")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('value')
    def validate_value(cls, v, values):
        """Validate sensor value based on type."""
        if 'sensor_type' in values:
            sensor_type = values['sensor_type']
            if sensor_type == SensorType.THERMAL and (v < -273 or v > 3000):
                raise ValueError('Temperature must be between -273°C and 3000°C')
            elif sensor_type == SensorType.PRESSURE and v < 0:
                raise ValueError('Pressure must be positive')
        return v
    
    @model_validator(mode='after')
    def validate_quality_consistency(self):
        """Validate quality score consistency with value."""
        if self.quality_score < 50 and abs(self.value) > 1000:
            raise ValueError('High values with low quality scores may indicate sensor issues')
        return self
    
    def to_cassandra_dict(self) -> Dict[str, Any]:
        """Convert model to Cassandra-compatible dictionary."""
        data = self.model_dump()
        # Serialize metadata to JSON string
        if 'metadata' in data:
            data['metadata'] = json.dumps(data['metadata']) if data['metadata'] else '{}'
        return data
    
    @classmethod
    def from_cassandra_dict(cls, data: Dict[str, Any]) -> 'SensorReading':
        """Create model from Cassandra dictionary."""
        # Deserialize metadata from JSON string
        if 'metadata' in data and isinstance(data['metadata'], str):
            data['metadata'] = safe_json_loads_with_fallback(data['metadata'], 'metadata', 5000, {})
        return cls(**data)


class ProcessMonitoring(BaseTimeSeriesModel):
    """Process monitoring event model."""
    
    # Primary identifiers
    process_id: str = Field(..., description="Process identifier")
    machine_id: str = Field(..., description="Machine identifier")
    operator_id: Optional[str] = Field(None, description="Operator identifier")
    
    # Event data
    event_type: str = Field(..., description="Type of monitoring event")
    event_data: Dict[str, Any] = Field(default_factory=dict, description="Event-specific data")
    severity: AlertSeverity = Field(default=AlertSeverity.LOW, description="Event severity")
    
    # Process context
    process_status: ProcessStatus = Field(..., description="Current process status")
    layer_number: Optional[int] = Field(None, ge=1, description="Current layer number")
    progress_percentage: float = Field(0.0, ge=0, le=100, description="Process progress percentage")
    
    # Metadata
    session_id: Optional[str] = Field(None, description="Session identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('event_type')
    def validate_event_type(cls, v):
        """Validate event type."""
        valid_types = [
            'LAYER_START', 'LAYER_COMPLETE', 'POWDER_DEPOSIT', 'LASER_FIRE',
            'TEMPERATURE_ALERT', 'PRESSURE_ALERT', 'QUALITY_CHECK', 'SYSTEM_ERROR',
            'MAINTENANCE_REQUIRED', 'CALIBRATION', 'USER_ACTION'
        ]
        if v not in valid_types:
            raise ValueError(f'Event type must be one of: {", ".join(valid_types)}')
        return v
    
    def to_cassandra_dict(self) -> Dict[str, Any]:
        """Convert model to Cassandra-compatible dictionary."""
        data = self.model_dump()
        # Serialize JSON fields to strings
        for field in ['event_data', 'metadata']:
            if field in data:
                data[field] = json.dumps(data[field]) if data[field] else '{}'
        return data
    
    @classmethod
    def from_cassandra_dict(cls, data: Dict[str, Any]) -> 'ProcessMonitoring':
        """Create model from Cassandra dictionary."""
        # Deserialize JSON fields from strings
        for field in ['event_data', 'metadata']:
            if field in data and isinstance(data[field], str):
                data[field] = safe_json_loads_with_fallback(data[field], field, 5000, {})
        return cls(**data)


class MachineStatusUpdate(BaseTimeSeriesModel):
    """Machine status update model."""
    
    # Primary identifiers
    machine_id: str = Field(..., description="Machine identifier")
    operator_id: Optional[str] = Field(None, description="Operator identifier")
    
    # Status data
    status: MachineStatus = Field(..., description="Current machine status")
    previous_status: Optional[MachineStatus] = Field(None, description="Previous machine status")
    
    # Performance metrics
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")
    health_score: float = Field(..., ge=0, le=100, description="Machine health score (0-100)")
    
    # Alerts and issues
    active_alerts: List[str] = Field(default_factory=list, description="Active alert IDs")
    maintenance_required: bool = Field(default=False, description="Maintenance required flag")
    next_maintenance_date: Optional[datetime] = Field(None, description="Next scheduled maintenance")
    
    # Metadata
    location: Optional[str] = Field(None, description="Machine location")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @model_validator(mode='after')
    def validate_status_transition(self):
        """Validate status transition logic."""
        if self.previous_status and self.status == self.previous_status:
            raise ValueError('Status must change for status update')
        return self
    
    def to_cassandra_dict(self) -> Dict[str, Any]:
        """Convert model to Cassandra-compatible dictionary."""
        data = self.model_dump()
        # Serialize JSON fields to strings
        for field in ['performance_metrics', 'metadata']:
            if field in data:
                data[field] = json.dumps(data[field]) if data[field] else '{}'
        return data
    
    @classmethod
    def from_cassandra_dict(cls, data: Dict[str, Any]) -> 'MachineStatusUpdate':
        """Create model from Cassandra dictionary."""
        # Deserialize JSON fields from strings
        for field in ['performance_metrics', 'metadata']:
            if field in data and isinstance(data[field], str):
                data[field] = safe_json_loads_with_fallback(data[field], field, 5000, {})
        return cls(**data)


class AnalyticsAggregation(BaseTimeSeriesModel):
    """Analytics aggregation model for pre-computed metrics."""
    
    # Primary identifiers
    aggregation_id: str = Field(..., description="Unique aggregation identifier")
    process_id: Optional[str] = Field(None, description="Process identifier")
    machine_id: Optional[str] = Field(None, description="Machine identifier")
    sensor_id: Optional[str] = Field(None, description="Sensor identifier")
    
    # Aggregation metadata
    aggregation_type: str = Field(..., description="Type of aggregation")
    time_window: str = Field(..., description="Time window (e.g., '1h', '1d', '1w')")
    granularity: str = Field(..., description="Data granularity")
    
    # Aggregated metrics
    count: int = Field(..., ge=0, description="Number of data points")
    min_value: float = Field(..., description="Minimum value")
    max_value: float = Field(..., description="Maximum value")
    avg_value: float = Field(..., description="Average value")
    median_value: Optional[float] = Field(None, description="Median value")
    std_dev: Optional[float] = Field(None, description="Standard deviation")
    percentiles: Dict[str, float] = Field(default_factory=dict, description="Percentile values")
    
    # Quality metrics
    data_quality_score: float = Field(..., ge=0, le=100, description="Overall data quality score")
    missing_data_points: int = Field(0, ge=0, description="Number of missing data points")
    
    # Metadata
    calculation_method: str = Field(default="standard", description="Calculation method used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('aggregation_type')
    def validate_aggregation_type(cls, v):
        """Validate aggregation type."""
        valid_types = [
            'HOURLY', 'DAILY', 'WEEKLY', 'MONTHLY', 'REALTIME',
            'SENSOR_AGGREGATION', 'PROCESS_AGGREGATION', 'MACHINE_AGGREGATION'
        ]
        if v not in valid_types:
            raise ValueError(f'Aggregation type must be one of: {", ".join(valid_types)}')
        return v
    
    def to_cassandra_dict(self) -> Dict[str, Any]:
        """Convert model to Cassandra-compatible dictionary."""
        data = self.model_dump()
        # Serialize JSON fields to strings
        for field in ['percentiles', 'metadata']:
            if field in data:
                data[field] = json.dumps(data[field]) if data[field] else '{}'
        return data
    
    @classmethod
    def from_cassandra_dict(cls, data: Dict[str, Any]) -> 'AnalyticsAggregation':
        """Create model from Cassandra dictionary."""
        # Deserialize JSON fields from strings
        for field in ['percentiles', 'metadata']:
            if field in data and isinstance(data[field], str):
                data[field] = safe_json_loads_with_fallback(data[field], field, 5000, {})
        return cls(**data)


class AlertEvent(BaseTimeSeriesModel):
    """Alert event model for system notifications."""
    
    # Primary identifiers
    alert_id: str = Field(..., description="Unique alert identifier")
    process_id: Optional[str] = Field(None, description="Process identifier")
    machine_id: Optional[str] = Field(None, description="Machine identifier")
    sensor_id: Optional[str] = Field(None, description="Sensor identifier")
    
    # Alert data
    alert_type: str = Field(..., description="Type of alert")
    severity: AlertSeverity = Field(..., description="Alert severity level")
    title: str = Field(..., description="Alert title")
    description: str = Field(..., description="Alert description")
    
    # Alert context
    threshold_value: Optional[float] = Field(None, description="Threshold value that triggered alert")
    actual_value: Optional[float] = Field(None, description="Actual value that triggered alert")
    tolerance: Optional[float] = Field(None, description="Tolerance range")
    
    # Alert lifecycle
    status: str = Field(default="ACTIVE", description="Alert status")
    acknowledged: bool = Field(default=False, description="Alert acknowledged flag")
    acknowledged_by: Optional[str] = Field(None, description="User who acknowledged alert")
    acknowledged_at: Optional[datetime] = Field(None, description="Acknowledgment timestamp")
    resolved: bool = Field(default=False, description="Alert resolved flag")
    resolved_at: Optional[datetime] = Field(None, description="Resolution timestamp")
    
    # Metadata
    source_system: str = Field(..., description="System that generated the alert")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('alert_type')
    def validate_alert_type(cls, v):
        """Validate alert type."""
        valid_types = [
            'TEMPERATURE_HIGH', 'TEMPERATURE_LOW', 'PRESSURE_HIGH', 'PRESSURE_LOW',
            'QUALITY_DEGRADATION', 'MACHINE_ERROR', 'SENSOR_FAILURE', 'MAINTENANCE_DUE',
            'PROCESS_DEVIATION', 'SYSTEM_OVERLOAD', 'CONNECTIVITY_ISSUE'
        ]
        if v not in valid_types:
            raise ValueError(f'Alert type must be one of: {", ".join(valid_types)}')
        return v
    
    @model_validator(mode='after')
    def validate_alert_lifecycle(self):
        """Validate alert lifecycle logic."""
        if self.resolved and not self.acknowledged:
            raise ValueError('Alert cannot be resolved without being acknowledged')
        if self.acknowledged and not self.acknowledged_by:
            raise ValueError('Acknowledged alert must have acknowledged_by field')
        return self
    
    def to_cassandra_dict(self) -> Dict[str, Any]:
        """Convert model to Cassandra-compatible dictionary."""
        data = self.model_dump()
        # Serialize metadata to JSON string
        if 'metadata' in data:
            data['metadata'] = json.dumps(data['metadata']) if data['metadata'] else '{}'
        return data
    
    @classmethod
    def from_cassandra_dict(cls, data: Dict[str, Any]) -> 'AlertEvent':
        """Create model from Cassandra dictionary."""
        # Deserialize metadata from JSON string
        if 'metadata' in data and isinstance(data['metadata'], str):
            data['metadata'] = safe_json_loads_with_fallback(data['metadata'], 'metadata', 5000, {})
        return cls(**data)


class TimeSeriesQuery(BaseModel):
    """Time-series query model for data retrieval."""
    
    # Query parameters
    start_time: datetime = Field(..., description="Query start time")
    end_time: datetime = Field(..., description="Query end time")
    
    # Filters
    sensor_ids: Optional[List[str]] = Field(None, description="Filter by sensor IDs")
    process_ids: Optional[List[str]] = Field(None, description="Filter by process IDs")
    machine_ids: Optional[List[str]] = Field(None, description="Filter by machine IDs")
    sensor_types: Optional[List[SensorType]] = Field(None, description="Filter by sensor types")
    
    # Aggregation
    aggregation: Optional[str] = Field(None, description="Aggregation function")
    granularity: Optional[str] = Field(None, description="Time granularity")
    
    # Pagination
    limit: Optional[int] = Field(None, ge=1, le=10000, description="Maximum number of results")
    offset: Optional[int] = Field(None, ge=0, description="Number of results to skip")
    
    # Sorting
    order_by: Optional[str] = Field("timestamp", description="Field to order by")
    order_direction: Optional[str] = Field("ASC", description="Order direction")
    
    @validator('end_time')
    def validate_time_range(cls, v, values):
        """Validate time range."""
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('End time must be after start time')
        return v
    
    @validator('aggregation')
    def validate_aggregation(cls, v):
        """Validate aggregation function."""
        if v is not None:
            valid_aggregations = ['AVG', 'SUM', 'MIN', 'MAX', 'COUNT', 'STDDEV']
            if v not in valid_aggregations:
                raise ValueError(f'Aggregation must be one of: {", ".join(valid_aggregations)}')
        return v
