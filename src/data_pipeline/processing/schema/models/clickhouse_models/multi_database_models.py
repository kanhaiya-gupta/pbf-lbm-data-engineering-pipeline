"""
ClickHouse Multi-Database Integration Models

This module contains Pydantic models for multi-database data integrated into ClickHouse data warehouse.
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator


# ============================================================================
# ENUMS
# ============================================================================

class CacheOperation(str, Enum):
    """Cache operation enumeration."""
    GET = "get"
    SET = "set"
    DELETE = "delete"
    UPDATE = "update"
    INCREMENT = "increment"
    DECREMENT = "decrement"

class CacheType(str, Enum):
    """Cache type enumeration."""
    SESSION = "session"
    DATA = "data"
    ANALYTICS = "analytics"
    TEMPORARY = "temporary"
    PERSISTENT = "persistent"

class CompressionType(str, Enum):
    """Compression type enumeration."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    SNAPPY = "snappy"
    ZSTD = "zstd"

class DataType(str, Enum):
    """Data type enumeration."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    JSON = "json"
    BINARY = "binary"

class JobType(str, Enum):
    """Job type enumeration."""
    BACKUP = "backup"
    RESTORE = "restore"
    ANALYSIS = "analysis"
    REPORT = "report"
    MAINTENANCE = "maintenance"
    CALIBRATION = "calibration"

class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class ExecutionStatus(str, Enum):
    """Execution status enumeration."""
    STARTED = "started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class SessionStatus(str, Enum):
    """Session status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    SUSPENDED = "suspended"

class SessionType(str, Enum):
    """Session type enumeration."""
    WEB = "web"
    API = "api"
    MOBILE = "mobile"
    DESKTOP = "desktop"
    SYSTEM = "system"

class LoginMethod(str, Enum):
    """Login method enumeration."""
    PASSWORD = "password"
    TOKEN = "token"
    OAUTH = "oauth"
    SSO = "sso"
    API_KEY = "api_key"

class SecurityLevel(str, Enum):
    """Security level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AggregationType(str, Enum):
    """Aggregation type enumeration."""
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    MEDIAN = "median"
    PERCENTILE = "percentile"

class AggregationWindow(str, Enum):
    """Aggregation window enumeration."""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"

class MonitoringType(str, Enum):
    """Monitoring type enumeration."""
    SENSOR = "sensor"
    PROCESS = "process"
    MACHINE = "machine"
    ENVIRONMENTAL = "environmental"
    QUALITY = "quality"

class ISPMAnomalyType(str, Enum):
    """ISPM anomaly type enumeration."""
    TEMPERATURE_SPIKE = "temperature_spike"
    PRESSURE_DROP = "pressure_drop"
    VIBRATION_INCREASE = "vibration_increase"
    POWER_FLUCTUATION = "power_fluctuation"
    FLOW_ANOMALY = "flow_anomaly"

class AnomalySeverity(str, Enum):
    """Anomaly severity enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# COMPONENT MODELS
# ============================================================================

class CachePerformanceMetrics(BaseModel):
    """Cache performance metrics model."""
    response_time: float = Field(..., ge=0, description="Response time in milliseconds")
    memory_usage: int = Field(..., ge=0, description="Memory usage in bytes")
    cpu_usage: float = Field(..., ge=0, le=100, description="CPU usage percentage")
    network_latency: float = Field(..., ge=0, description="Network latency in milliseconds")

class CacheAnalytics(BaseModel):
    """Cache analytics model."""
    access_frequency: float = Field(..., ge=0, description="Access frequency")
    hit_rate: float = Field(..., ge=0, le=1, description="Cache hit rate")
    miss_rate: float = Field(..., ge=0, le=1, description="Cache miss rate")
    eviction_count: int = Field(..., ge=0, description="Eviction count")

class QueueAnalytics(BaseModel):
    """Queue analytics model."""
    queue_position: int = Field(..., ge=0, description="Queue position")
    queue_priority: int = Field(..., ge=0, description="Queue priority")
    estimated_completion_time: Optional[datetime] = Field(None, description="Estimated completion time")
    actual_completion_time: Optional[datetime] = Field(None, description="Actual completion time")

class ErrorInfo(BaseModel):
    """Error information model."""
    error_count: int = Field(..., ge=0, description="Error count")
    error_messages: List[str] = Field(..., description="Error messages")
    last_error: Optional[str] = Field(None, description="Last error message")
    retry_count: int = Field(..., ge=0, description="Retry count")

class SessionAnalytics(BaseModel):
    """Session analytics model."""
    page_views: int = Field(..., ge=0, description="Page views")
    api_calls: int = Field(..., ge=0, description="API calls")
    data_queries: int = Field(..., ge=0, description="Data queries")
    file_downloads: int = Field(..., ge=0, description="File downloads")
    session_actions: int = Field(..., ge=0, description="Session actions")

class SecurityInfo(BaseModel):
    """Security information model."""
    security_level: SecurityLevel = Field(..., description="Security level")
    risk_score: float = Field(..., ge=0, le=1, description="Risk score")
    suspicious_activity: bool = Field(False, description="Suspicious activity flag")
    failed_attempts: int = Field(..., ge=0, description="Failed login attempts")
    last_password_change: Optional[datetime] = Field(None, description="Last password change")

class CassandraCalibrationInfo(BaseModel):
    """Cassandra calibration information model."""
    calibration_factor: float = Field(..., description="Calibration factor")
    calibration_accuracy: float = Field(..., ge=0, le=100, description="Calibration accuracy")
    calibration_uncertainty: float = Field(..., ge=0, description="Calibration uncertainty")
    last_calibrated: Optional[datetime] = Field(None, description="Last calibration timestamp")

class TimeSeriesAnalytics(BaseModel):
    """Time series analytics model."""
    min_value: float = Field(..., description="Minimum value")
    max_value: float = Field(..., description="Maximum value")
    avg_value: float = Field(..., description="Average value")
    sum_value: float = Field(..., description="Sum value")
    count_value: int = Field(..., ge=0, description="Count value")
    std_dev: float = Field(..., ge=0, description="Standard deviation")

class ISPMEnvironmentalConditions(BaseModel):
    """ISPM environmental conditions model."""
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Humidity percentage")
    pressure: float = Field(..., ge=0, description="Pressure in bar")
    vibration: float = Field(..., ge=0, description="Vibration level")
    noise_level: float = Field(..., ge=0, description="Noise level in dB")

class ISPMProcessParameters(BaseModel):
    """ISPM process parameters model."""
    process_speed: float = Field(..., ge=0, description="Process speed in mm/s")
    process_pressure: float = Field(..., ge=0, description="Process pressure in bar")
    process_temperature: float = Field(..., description="Process temperature in Celsius")
    process_flow_rate: float = Field(..., ge=0, description="Process flow rate in L/min")

class ISPMAnomalyDetection(BaseModel):
    """ISPM anomaly detection model."""
    anomaly_type: ISPMAnomalyType = Field(..., description="Anomaly type")
    anomaly_severity: AnomalySeverity = Field(..., description="Anomaly severity")
    anomaly_confidence: float = Field(..., ge=0, le=1, description="Anomaly confidence")
    anomaly_description: str = Field(..., description="Anomaly description")


# ============================================================================
# MAIN MODELS
# ============================================================================

class RedisCacheDataModel(BaseModel):
    """Redis Cache Data model for ClickHouse - matches redis_cache_data.sql schema with optional fields."""
    # Primary identifiers
    id: int = Field(..., description="Primary key ID")
    cache_key: str = Field(..., description="Cache key")
    cache_type: Optional[str] = Field(None, description="Cache type")
    timestamp: datetime = Field(..., description="Cache timestamp")
    
    # Cache metadata (flattened)
    cache_operation: Optional[str] = Field(None, description="Cache operation")
    cache_hit: Optional[int] = Field(None, ge=0, le=1, description="Cache hit flag (0/1)")
    cache_miss: Optional[int] = Field(None, ge=0, le=1, description="Cache miss flag (0/1)")
    cache_ttl: Optional[int] = Field(None, ge=0, description="Cache TTL in seconds")
    cache_size: Optional[int] = Field(None, ge=0, description="Cache size in bytes")
    cache_compression: Optional[str] = Field(None, description="Cache compression type")
    
    # Performance metrics (flattened)
    response_time: Optional[float] = Field(None, ge=0, description="Response time in milliseconds")
    memory_usage: Optional[int] = Field(None, ge=0, description="Memory usage in bytes")
    cpu_usage: Optional[float] = Field(None, ge=0, le=100, description="CPU usage percentage")
    network_latency: Optional[float] = Field(None, ge=0, description="Network latency in milliseconds")
    
    # Data information (flattened)
    data_type: Optional[str] = Field(None, description="Data type")
    data_size: Optional[int] = Field(None, ge=0, description="Data size in bytes")
    data_format: Optional[str] = Field(None, description="Data format")
    data_compression_ratio: Optional[float] = Field(None, ge=0, description="Data compression ratio")
    
    # User and session data (flattened)
    user_id: Optional[str] = Field(None, description="User ID")
    session_id: Optional[str] = Field(None, description="Session ID")
    client_ip: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="User agent")
    
    # Cache analytics (flattened)
    access_frequency: Optional[int] = Field(None, ge=0, description="Access frequency")
    last_accessed: Optional[datetime] = Field(None, description="Last accessed timestamp")
    expiration_time: Optional[datetime] = Field(None, description="Expiration timestamp")
    cache_priority: Optional[int] = Field(None, ge=0, description="Cache priority")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class JobQueueDataModel(BaseModel):
    """Job Queue Data model for ClickHouse - matches job_queue_data.sql schema with optional fields."""
    # Primary identifiers
    id: int = Field(..., description="Primary key ID")
    job_id: str = Field(..., description="Job ID")
    cache_key: Optional[str] = Field(None, description="Cache key")
    timestamp: datetime = Field(..., description="Job timestamp")
    
    # Job information (flattened)
    job_type: Optional[str] = Field(None, description="Job type")
    job_status: Optional[str] = Field(None, description="Job status")
    job_priority: Optional[int] = Field(None, ge=0, description="Job priority")
    job_category: Optional[str] = Field(None, description="Job category")
    job_description: Optional[str] = Field(None, description="Job description")
    
    # Process and machine context (flattened)
    process_id: Optional[str] = Field(None, description="Process ID")
    machine_id: Optional[str] = Field(None, description="Machine ID")
    build_id: Optional[str] = Field(None, description="Build ID")
    part_id: Optional[str] = Field(None, description="Part ID")
    
    # Job execution data (flattened)
    execution_status: Optional[str] = Field(None, description="Execution status")
    execution_start_time: Optional[datetime] = Field(None, description="Execution start time")
    execution_end_time: Optional[datetime] = Field(None, description="Execution end time")
    execution_duration: Optional[float] = Field(None, ge=0, description="Execution duration in seconds")
    execution_attempts: Optional[int] = Field(None, ge=0, description="Execution attempts")
    max_attempts: Optional[int] = Field(None, ge=0, description="Maximum attempts")
    
    # Performance metrics (flattened)
    cpu_usage: Optional[float] = Field(None, ge=0, le=100, description="CPU usage percentage")
    memory_usage: Optional[int] = Field(None, ge=0, description="Memory usage in bytes")
    disk_usage: Optional[int] = Field(None, ge=0, description="Disk usage in bytes")
    network_usage: Optional[int] = Field(None, ge=0, description="Network usage in bytes")
    processing_time: Optional[float] = Field(None, ge=0, description="Processing time in seconds")
    queue_wait_time: Optional[float] = Field(None, ge=0, description="Queue wait time in seconds")
    
    # Job parameters (flattened)
    job_parameters: Optional[str] = Field(None, description="Job parameters")
    job_input_data: Optional[str] = Field(None, description="Job input data")
    job_output_data: Optional[str] = Field(None, description="Job output data")
    job_dependencies: Optional[List[str]] = Field(None, description="Job dependencies")
    job_requirements: Optional[str] = Field(None, description="Job requirements")
    
    # User and session data (flattened)
    user_id: Optional[str] = Field(None, description="User ID")
    session_id: Optional[str] = Field(None, description="Session ID")
    created_by: Optional[str] = Field(None, description="Created by")
    assigned_to: Optional[str] = Field(None, description="Assigned to")
    
    # Queue analytics (flattened)
    queue_position: Optional[int] = Field(None, ge=0, description="Queue position")
    queue_priority: Optional[int] = Field(None, ge=0, description="Queue priority")
    estimated_completion_time: Optional[datetime] = Field(None, description="Estimated completion time")
    actual_completion_time: Optional[datetime] = Field(None, description="Actual completion time")
    
    # Error handling (flattened)
    error_count: Optional[int] = Field(None, ge=0, description="Error count")
    error_messages: Optional[List[str]] = Field(None, description="Error messages")
    last_error: Optional[str] = Field(None, description="Last error message")
    retry_count: Optional[int] = Field(None, ge=0, description="Retry count")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class UserSessionDataModel(BaseModel):
    """User Session Data model for ClickHouse - matches user_session_data.sql schema with optional fields."""
    # Primary identifiers
    id: int = Field(..., description="Primary key ID")
    session_id: str = Field(..., description="Session ID")
    cache_key: Optional[str] = Field(None, description="Cache key")
    timestamp: datetime = Field(..., description="Session timestamp")
    
    # User information (flattened)
    user_id: Optional[str] = Field(None, description="User ID")
    username: Optional[str] = Field(None, description="Username")
    user_role: Optional[str] = Field(None, description="User role")
    user_permissions: Optional[List[str]] = Field(None, description="User permissions")
    user_department: Optional[str] = Field(None, description="User department")
    user_team: Optional[str] = Field(None, description="User team")
    
    # Session information (flattened)
    session_status: Optional[str] = Field(None, description="Session status")
    session_type: Optional[str] = Field(None, description="Session type")
    session_duration: Optional[float] = Field(None, ge=0, description="Session duration in seconds")
    session_start_time: Optional[datetime] = Field(None, description="Session start time")
    session_end_time: Optional[datetime] = Field(None, description="Session end time")
    last_activity: Optional[datetime] = Field(None, description="Last activity timestamp")
    
    # Authentication data (flattened)
    login_method: Optional[str] = Field(None, description="Login method")
    login_ip: Optional[str] = Field(None, description="Login IP address")
    login_location: Optional[str] = Field(None, description="Login location")
    login_device: Optional[str] = Field(None, description="Login device")
    login_browser: Optional[str] = Field(None, description="Login browser")
    login_os: Optional[str] = Field(None, description="Login operating system")
    
    # Session analytics (flattened)
    page_views: Optional[int] = Field(None, ge=0, description="Page views count")
    api_calls: Optional[int] = Field(None, ge=0, description="API calls count")
    data_queries: Optional[int] = Field(None, ge=0, description="Data queries count")
    file_downloads: Optional[int] = Field(None, ge=0, description="File downloads count")
    session_actions: Optional[int] = Field(None, ge=0, description="Session actions count")
    
    # Performance metrics (flattened)
    response_time: Optional[float] = Field(None, ge=0, description="Response time in milliseconds")
    memory_usage: Optional[int] = Field(None, ge=0, description="Memory usage in bytes")
    cpu_usage: Optional[float] = Field(None, ge=0, le=100, description="CPU usage percentage")
    network_usage: Optional[int] = Field(None, ge=0, description="Network usage in bytes")
    
    # Security data (flattened)
    security_level: Optional[str] = Field(None, description="Security level")
    risk_score: Optional[float] = Field(None, ge=0, le=100, description="Risk score")
    suspicious_activity: Optional[int] = Field(None, ge=0, le=1, description="Suspicious activity flag (0/1)")
    failed_attempts: Optional[int] = Field(None, ge=0, description="Failed login attempts")
    last_password_change: Optional[datetime] = Field(None, description="Last password change timestamp")
    
    # Session context (flattened)
    active_processes: Optional[List[str]] = Field(None, description="Active processes")
    active_machines: Optional[List[str]] = Field(None, description="Active machines")
    active_builds: Optional[List[str]] = Field(None, description="Active builds")
    current_workspace: Optional[str] = Field(None, description="Current workspace")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class CassandraTimeSeriesModel(BaseModel):
    """Cassandra Time Series model for ClickHouse - matches cassandra_time_series.sql schema with optional fields."""
    # Primary identifiers
    id: int = Field(..., description="Primary key ID")
    sensor_id: str = Field(..., description="Sensor ID")
    process_id: Optional[str] = Field(None, description="Process ID")
    build_id: Optional[str] = Field(None, description="Build ID")
    machine_id: Optional[str] = Field(None, description="Machine ID")
    timestamp: datetime = Field(..., description="Time series timestamp")
    
    # Time series data (flattened)
    sensor_type: Optional[str] = Field(None, description="Sensor type")
    value: Optional[float] = Field(None, description="Sensor value")
    unit: Optional[str] = Field(None, description="Measurement unit")
    location: Optional[str] = Field(None, description="Sensor location")
    status: Optional[str] = Field(None, description="Sensor status")
    quality_score: Optional[float] = Field(None, ge=0, le=100, description="Quality score")
    
    # Aggregation data (flattened)
    aggregation_type: Optional[str] = Field(None, description="Aggregation type")
    aggregation_window: Optional[str] = Field(None, description="Aggregation window")
    min_value: Optional[float] = Field(None, description="Minimum value")
    max_value: Optional[float] = Field(None, description="Maximum value")
    avg_value: Optional[float] = Field(None, description="Average value")
    sum_value: Optional[float] = Field(None, description="Sum value")
    count_value: Optional[int] = Field(None, ge=0, description="Count value")
    std_dev: Optional[float] = Field(None, ge=0, description="Standard deviation")
    
    # Calibration data (flattened)
    calibration_factor: Optional[float] = Field(None, ge=0, description="Calibration factor")
    calibration_accuracy: Optional[float] = Field(None, ge=0, le=100, description="Calibration accuracy")
    calibration_uncertainty: Optional[float] = Field(None, ge=0, description="Calibration uncertainty")
    last_calibrated: Optional[datetime] = Field(None, description="Last calibrated timestamp")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ISPMMonitoringModel(BaseModel):
    """ISPM Monitoring model for ClickHouse - matches ispm_monitoring.sql schema with optional fields."""
    # Primary identifiers
    id: int = Field(..., description="Primary key ID")
    monitoring_id: str = Field(..., description="Monitoring ID")
    process_id: Optional[str] = Field(None, description="Process ID")
    build_id: Optional[str] = Field(None, description="Build ID")
    machine_id: Optional[str] = Field(None, description="Machine ID")
    timestamp: datetime = Field(..., description="Monitoring timestamp")
    
    # Monitoring data (flattened)
    monitoring_type: Optional[str] = Field(None, description="Monitoring type")
    sensor_id: Optional[str] = Field(None, description="Sensor ID")
    value: Optional[float] = Field(None, description="Sensor value")
    unit: Optional[str] = Field(None, description="Measurement unit")
    location: Optional[str] = Field(None, description="Sensor location")
    status: Optional[str] = Field(None, description="Sensor status")
    
    # Anomaly detection (flattened)
    anomaly_score: Optional[float] = Field(None, ge=0, le=1, description="Anomaly score")
    anomaly_type: Optional[str] = Field(None, description="Anomaly type")
    anomaly_severity: Optional[str] = Field(None, description="Anomaly severity")
    anomaly_confidence: Optional[float] = Field(None, ge=0, le=1, description="Anomaly confidence")
    anomaly_detection_method: Optional[str] = Field(None, description="Anomaly detection method")
    
    # Quality metrics (flattened)
    quality_score: Optional[float] = Field(None, ge=0, le=100, description="Quality score")
    data_quality: Optional[str] = Field(None, description="Data quality")
    measurement_uncertainty: Optional[float] = Field(None, ge=0, description="Measurement uncertainty")
    calibration_status: Optional[str] = Field(None, description="Calibration status")
    
    # Environmental conditions (flattened)
    temperature: Optional[float] = Field(None, description="Temperature")
    humidity: Optional[float] = Field(None, ge=0, le=100, description="Humidity percentage")
    pressure: Optional[float] = Field(None, ge=0, description="Pressure")
    vibration: Optional[float] = Field(None, ge=0, description="Vibration level")
    noise_level: Optional[float] = Field(None, ge=0, description="Noise level")
    
    # Process parameters (flattened)
    process_speed: Optional[float] = Field(None, ge=0, description="Process speed")
    process_pressure: Optional[float] = Field(None, ge=0, description="Process pressure")
    process_temperature: Optional[float] = Field(None, description="Process temperature")
    process_flow_rate: Optional[float] = Field(None, ge=0, description="Process flow rate")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
