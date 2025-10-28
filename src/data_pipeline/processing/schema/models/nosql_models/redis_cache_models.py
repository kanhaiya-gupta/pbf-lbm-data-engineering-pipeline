"""
Redis Cache Models for PBF-LB/M Data Pipeline

This module defines Pydantic models for structured Redis caching,
optimized for high-performance manufacturing data operations.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
import json


class CacheStatus(str, Enum):
    """Cache status enumeration."""
    ACTIVE = "active"
    EXPIRED = "expired"
    INVALID = "invalid"
    PENDING = "pending"


class ProcessStatus(str, Enum):
    """Process status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


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


class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SessionStatus(str, Enum):
    """Session status enumeration."""
    ACTIVE = "active"
    EXPIRED = "expired"
    TERMINATED = "terminated"


# Base Redis Cache Model
class BaseRedisCache(BaseModel):
    """Base model for Redis cache entries."""
    
    cache_key: str = Field(..., description="Redis cache key")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Cache creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Cache last update timestamp")
    ttl: Optional[int] = Field(None, description="Time to live in seconds")
    status: CacheStatus = Field(default=CacheStatus.ACTIVE, description="Cache status")
    version: int = Field(default=1, description="Cache version")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        use_enum_values = True


# Process Data Cache Models
class ProcessDataCache(BaseRedisCache):
    """Process data cache model."""
    
    process_id: str = Field(..., description="Process identifier")
    build_id: str = Field(..., description="Build identifier")
    part_id: str = Field(..., description="Part identifier")
    machine_id: str = Field(..., description="Machine identifier")
    status: ProcessStatus = Field(..., description="Process status")
    
    # Process parameters
    temperature: Optional[float] = Field(None, description="Process temperature in Celsius")
    laser_power: Optional[float] = Field(None, description="Laser power in watts")
    scan_speed: Optional[float] = Field(None, description="Scan speed in mm/s")
    layer_thickness: Optional[float] = Field(None, description="Layer thickness in mm")
    
    # Quality metrics
    density: Optional[float] = Field(None, description="Part density")
    surface_roughness: Optional[float] = Field(None, description="Surface roughness in μm")
    dimensional_accuracy: Optional[float] = Field(None, description="Dimensional accuracy in %")
    
    # Metadata
    operator: Optional[str] = Field(None, description="Operator name")
    notes: Optional[str] = Field(None, description="Process notes")
    tags: List[str] = Field(default_factory=list, description="Process tags")
    
    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        if v is not None and (v < 0 or v > 3000):
            raise ValueError('Temperature must be between 0 and 3000°C')
        return v
    
    @field_validator('laser_power')
    @classmethod
    def validate_laser_power(cls, v):
        if v is not None and (v < 0 or v > 1000):
            raise ValueError('Laser power must be between 0 and 1000W')
        return v
    
    @field_validator('scan_speed')
    @classmethod
    def validate_scan_speed(cls, v):
        if v is not None and (v < 0 or v > 10000):
            raise ValueError('Scan speed must be between 0 and 10000 mm/s')
        return v


class MachineStatusCache(BaseRedisCache):
    """Machine status cache model."""
    
    machine_id: str = Field(..., description="Machine identifier")
    status: MachineStatus = Field(..., description="Machine status")
    
    # Environmental conditions
    temperature: Optional[float] = Field(None, description="Ambient temperature in Celsius")
    humidity: Optional[float] = Field(None, description="Humidity percentage")
    pressure: Optional[float] = Field(None, description="Atmospheric pressure in Pa")
    
    # Machine components
    laser_status: Optional[str] = Field(None, description="Laser system status")
    build_platform_status: Optional[str] = Field(None, description="Build platform status")
    powder_feeder_status: Optional[str] = Field(None, description="Powder feeder status")
    gas_system_status: Optional[str] = Field(None, description="Gas system status")
    
    # Operational data
    current_build_id: Optional[str] = Field(None, description="Current build identifier")
    total_operating_hours: Optional[float] = Field(None, description="Total operating hours")
    maintenance_due: Optional[bool] = Field(None, description="Maintenance due flag")
    
    # Metadata
    last_maintenance: Optional[datetime] = Field(None, description="Last maintenance date")
    next_maintenance: Optional[datetime] = Field(None, description="Next maintenance date")
    operator: Optional[str] = Field(None, description="Current operator")
    
    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        if v is not None and (v < -50 or v > 100):
            raise ValueError('Temperature must be between -50 and 100°C')
        return v
    
    @field_validator('humidity')
    @classmethod
    def validate_humidity(cls, v):
        if v is not None and (v < 0 or v > 100):
            raise ValueError('Humidity must be between 0 and 100%')
        return v


class SensorReadingCache(BaseRedisCache):
    """Sensor reading cache model."""
    
    sensor_id: str = Field(..., description="Sensor identifier")
    sensor_type: SensorType = Field(..., description="Sensor type")
    
    # Reading data
    value: float = Field(..., description="Sensor reading value")
    unit: str = Field(..., description="Measurement unit")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Reading timestamp")
    
    # Sensor metadata
    sensor_location: Optional[str] = Field(None, description="Sensor location")
    calibration_date: Optional[datetime] = Field(None, description="Last calibration date")
    accuracy: Optional[float] = Field(None, description="Sensor accuracy")
    
    # Quality indicators
    signal_quality: Optional[str] = Field(None, description="Signal quality assessment")
    noise_level: Optional[float] = Field(None, description="Noise level")
    data_quality: Optional[str] = Field(None, description="Data quality rating")
    
    @field_validator('value')
    @classmethod
    def validate_value(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError('Value must be numeric')
        return float(v)
    
    @field_validator('accuracy')
    @classmethod
    def validate_accuracy(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError('Accuracy must be between 0 and 1')
        return v


class AnalyticsCache(BaseRedisCache):
    """Analytics data cache model."""
    
    analytics_type: str = Field(..., description="Type of analytics (daily, weekly, monthly)")
    date: str = Field(..., description="Analytics date")
    
    # Process metrics
    total_processes: int = Field(..., description="Total number of processes")
    successful_processes: int = Field(..., description="Number of successful processes")
    failed_processes: int = Field(..., description="Number of failed processes")
    success_rate: float = Field(..., description="Process success rate")
    
    # Performance metrics
    average_temperature: Optional[float] = Field(None, description="Average process temperature")
    average_laser_power: Optional[float] = Field(None, description="Average laser power")
    average_scan_speed: Optional[float] = Field(None, description="Average scan speed")
    total_build_time: Optional[float] = Field(None, description="Total build time in hours")
    
    # Quality metrics
    average_density: Optional[float] = Field(None, description="Average part density")
    average_surface_roughness: Optional[float] = Field(None, description="Average surface roughness")
    average_dimensional_accuracy: Optional[float] = Field(None, description="Average dimensional accuracy")
    
    # Machine utilization
    machine_utilization: Optional[Dict[str, float]] = Field(None, description="Machine utilization rates")
    operator_performance: Optional[Dict[str, float]] = Field(None, description="Operator performance metrics")
    
    @field_validator('success_rate')
    @classmethod
    def validate_success_rate(cls, v):
        if v < 0 or v > 1:
            raise ValueError('Success rate must be between 0 and 1')
        return v
    
    @model_validator(mode='after')
    def validate_process_counts(self):
        total = self.total_processes
        successful = self.successful_processes
        failed = self.failed_processes
        
        if successful + failed > total:
            raise ValueError('Successful + failed processes cannot exceed total processes')
        
        return self


class JobQueueCache(BaseRedisCache):
    """Job queue cache model."""
    
    job_id: str = Field(..., description="Job identifier")
    job_type: str = Field(..., description="Type of job")
    status: JobStatus = Field(..., description="Job status")
    priority: int = Field(default=5, description="Job priority (1-10, 1=highest)")
    
    # Job data
    process_id: Optional[str] = Field(None, description="Associated process ID")
    build_id: Optional[str] = Field(None, description="Associated build ID")
    machine_id: Optional[str] = Field(None, description="Target machine ID")
    
    # Job parameters
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Job parameters")
    estimated_duration: Optional[float] = Field(None, description="Estimated duration in minutes")
    required_resources: List[str] = Field(default_factory=list, description="Required resources")
    
    # Scheduling
    scheduled_at: Optional[datetime] = Field(None, description="Scheduled execution time")
    started_at: Optional[datetime] = Field(None, description="Actual start time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    
    # Results
    result_data: Optional[Dict[str, Any]] = Field(None, description="Job result data")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    @field_validator('priority')
    @classmethod
    def validate_priority(cls, v):
        if v < 1 or v > 10:
            raise ValueError('Priority must be between 1 and 10')
        return v
    
    @field_validator('estimated_duration')
    @classmethod
    def validate_duration(cls, v):
        if v is not None and v < 0:
            raise ValueError('Duration must be positive')
        return v


class UserSessionCache(BaseRedisCache):
    """User session cache model."""
    
    session_id: str = Field(..., description="Session identifier")
    user_id: str = Field(..., description="User identifier")
    status: SessionStatus = Field(..., description="Session status")
    
    # Session data
    username: str = Field(..., description="Username")
    role: str = Field(..., description="User role")
    permissions: List[str] = Field(default_factory=list, description="User permissions")
    
    # Session metadata
    ip_address: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="Client user agent")
    login_time: datetime = Field(default_factory=datetime.utcnow, description="Login timestamp")
    last_activity: datetime = Field(default_factory=datetime.utcnow, description="Last activity timestamp")
    
    # Session preferences
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    active_processes: List[str] = Field(default_factory=list, description="Active process IDs")
    favorite_machines: List[str] = Field(default_factory=list, description="Favorite machine IDs")
    
    @field_validator('ip_address')
    @classmethod
    def validate_ip_address(cls, v):
        if v is not None:
            import re
            ip_pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
            if not re.match(ip_pattern, v):
                raise ValueError('Invalid IP address format')
        return v


# Cache Statistics Model
class CacheStatistics(BaseModel):
    """Cache statistics model."""
    
    total_keys: int = Field(..., description="Total number of cache keys")
    memory_usage: int = Field(..., description="Memory usage in bytes")
    hit_rate: float = Field(..., description="Cache hit rate")
    miss_rate: float = Field(..., description="Cache miss rate")
    
    # Key type distribution
    string_keys: int = Field(default=0, description="Number of string keys")
    hash_keys: int = Field(default=0, description="Number of hash keys")
    list_keys: int = Field(default=0, description="Number of list keys")
    set_keys: int = Field(default=0, description="Number of set keys")
    sorted_set_keys: int = Field(default=0, description="Number of sorted set keys")
    
    # Performance metrics
    operations_per_second: float = Field(default=0, description="Operations per second")
    average_response_time: float = Field(default=0, description="Average response time in ms")
    
    # TTL distribution
    keys_with_ttl: int = Field(default=0, description="Keys with TTL")
    keys_without_ttl: int = Field(default=0, description="Keys without TTL")
    average_ttl: Optional[float] = Field(None, description="Average TTL in seconds")
    
    @field_validator('hit_rate')
    @classmethod
    def validate_hit_rate(cls, v):
        if v < 0 or v > 1:
            raise ValueError('Hit rate must be between 0 and 1')
        return v
    
    @field_validator('miss_rate')
    @classmethod
    def validate_miss_rate(cls, v):
        if v < 0 or v > 1:
            raise ValueError('Miss rate must be between 0 and 1')
        return v
    
    @model_validator(mode='after')
    def validate_rates(self):
        hit_rate = self.hit_rate
        miss_rate = self.miss_rate
        
        if abs(hit_rate + miss_rate - 1.0) > 0.01:  # Allow small floating point errors
            raise ValueError('Hit rate + miss rate must equal 1.0')
        
        return self


# Export all models
__all__ = [
    # Enums
    "CacheStatus",
    "ProcessStatus", 
    "MachineStatus",
    "SensorType",
    "JobStatus",
    "SessionStatus",
    
    # Base model
    "BaseRedisCache",
    
    # Cache models
    "ProcessDataCache",
    "MachineStatusCache", 
    "SensorReadingCache",
    "AnalyticsCache",
    "JobQueueCache",
    "UserSessionCache",
    "CacheStatistics"
]
