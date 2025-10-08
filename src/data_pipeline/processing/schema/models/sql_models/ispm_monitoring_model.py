"""
ISPM Monitoring Model

This module defines the Pydantic model for ISPM (In-Situ Process Monitoring) data.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import Field, validator, root_validator
from enum import Enum

from .base_model import BaseDataModel

class SensorType(str, Enum):
    """Enumeration of sensor types."""
    THERMAL = "THERMAL"
    OPTICAL = "OPTICAL"
    ACOUSTIC = "ACOUSTIC"
    VIBRATION = "VIBRATION"
    PRESSURE = "PRESSURE"
    GAS_ANALYSIS = "GAS_ANALYSIS"
    MELT_POOL = "MELT_POOL"
    LAYER_HEIGHT = "LAYER_HEIGHT"

class SignalQuality(str, Enum):
    """Enumeration of signal quality levels."""
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    FAIR = "FAIR"
    POOR = "POOR"
    UNKNOWN = "UNKNOWN"

class AnomalySeverity(str, Enum):
    """Enumeration of anomaly severity levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class SensorLocation(BaseDataModel):
    """Sensor location coordinates."""
    
    x_coordinate: float = Field(..., description="X coordinate in mm")
    y_coordinate: float = Field(..., description="Y coordinate in mm")
    z_coordinate: float = Field(..., description="Z coordinate in mm")
    
    def get_primary_key(self) -> str:
        return "id"
    
    def get_primary_key_value(self) -> Any:
        return getattr(self, "id", None)

class MeasurementRange(BaseDataModel):
    """Expected measurement range."""
    
    min_value: float = Field(..., description="Minimum expected value")
    max_value: float = Field(..., description="Maximum expected value")
    
    @validator('max_value')
    def validate_max_value(cls, v, values):
        """Validate that max_value is greater than min_value."""
        if 'min_value' in values and v <= values['min_value']:
            raise ValueError("max_value must be greater than min_value")
        return v
    
    def get_primary_key(self) -> str:
        return "id"
    
    def get_primary_key_value(self) -> Any:
        return getattr(self, "id", None)

class EnvironmentalConditions(BaseDataModel):
    """Environmental conditions during measurement."""
    
    temperature: Optional[float] = Field(None, description="Ambient temperature in Celsius")
    humidity: Optional[float] = Field(None, ge=0, le=100, description="Relative humidity percentage")
    vibration_level: Optional[float] = Field(None, ge=0, description="Vibration level in g")
    
    def get_primary_key(self) -> str:
        return "id"
    
    def get_primary_key_value(self) -> Any:
        return getattr(self, "id", None)

class ISPMMonitoringModel(BaseDataModel):
    """
    Pydantic model for ISPM (In-Situ Process Monitoring) data.
    
    This model represents sensor monitoring data for PBF-LB/M additive manufacturing,
    including sensor measurements, environmental conditions, and anomaly detection.
    """
    
    # Primary key and identifiers
    monitoring_id: str = Field(..., min_length=1, max_length=100, description="Unique identifier for the monitoring record")
    process_id: str = Field(..., min_length=1, max_length=100, description="Associated PBF process identifier")
    sensor_id: str = Field(..., min_length=1, max_length=50, description="Sensor identifier")
    
    # Monitoring timestamp
    timestamp: datetime = Field(..., description="Monitoring timestamp in ISO format")
    
    # Sensor information
    sensor_type: SensorType = Field(..., description="Type of sensor")
    sensor_location: Optional[SensorLocation] = Field(None, description="Sensor location coordinates")
    
    # Measurement data
    measurement_value: float = Field(..., description="Primary measurement value")
    unit: str = Field(..., min_length=1, max_length=20, description="Unit of measurement")
    measurement_range: Optional[MeasurementRange] = Field(None, description="Expected measurement range")
    measurement_accuracy: Optional[float] = Field(None, ge=0, description="Measurement accuracy/precision")
    sampling_rate: Optional[float] = Field(None, ge=0, le=1000000, description="Sampling rate in Hz")
    
    # Signal quality
    signal_quality: Optional[SignalQuality] = Field(None, description="Signal quality assessment")
    noise_level: Optional[float] = Field(None, ge=0, description="Noise level in the signal")
    
    # Calibration information
    calibration_status: Optional[bool] = Field(None, description="Sensor calibration status")
    last_calibration_date: Optional[datetime] = Field(None, description="Last calibration date")
    
    # Environmental conditions
    environmental_conditions: Optional[EnvironmentalConditions] = Field(None, description="Environmental conditions during measurement")
    
    # Anomaly detection
    anomaly_detected: Optional[bool] = Field(None, description="Whether an anomaly was detected")
    anomaly_type: Optional[str] = Field(None, min_length=1, max_length=100, description="Type of detected anomaly")
    anomaly_severity: Optional[AnomalySeverity] = Field(None, description="Severity of detected anomaly")
    
    # Additional data
    raw_data: Optional[bytes] = Field(None, description="Raw sensor data (if available)")
    processed_data: Dict[str, str] = Field(default_factory=dict, description="Processed sensor data as key-value pairs")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        schema_extra = {
            "example": {
                "monitoring_id": "MON_2024_001",
                "process_id": "PBF_2024_001",
                "sensor_id": "THERMAL_001",
                "timestamp": "2024-01-15T10:30:00Z",
                "sensor_type": "THERMAL",
                "sensor_location": {
                    "x_coordinate": 100.5,
                    "y_coordinate": 200.3,
                    "z_coordinate": 50.0
                },
                "measurement_value": 1200.5,
                "unit": "Celsius",
                "measurement_range": {
                    "min_value": 800.0,
                    "max_value": 1600.0
                },
                "measurement_accuracy": 0.1,
                "sampling_rate": 100.0,
                "signal_quality": "EXCELLENT",
                "noise_level": 0.05,
                "calibration_status": True,
                "last_calibration_date": "2024-01-01T00:00:00Z",
                "environmental_conditions": {
                    "temperature": 22.5,
                    "humidity": 45.0,
                    "vibration_level": 0.01
                },
                "anomaly_detected": False,
                "anomaly_type": None,
                "anomaly_severity": None,
                "processed_data": {
                    "filtered_value": "1200.3",
                    "trend": "stable"
                }
            }
        }
    
    @validator('monitoring_id')
    def validate_monitoring_id(cls, v):
        """Validate monitoring ID format."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Monitoring ID must contain only alphanumeric characters, underscores, and hyphens")
        return v
    
    @validator('sensor_id')
    def validate_sensor_id(cls, v):
        """Validate sensor ID format."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Sensor ID must contain only alphanumeric characters, underscores, and hyphens")
        return v
    
    @validator('measurement_value')
    def validate_measurement_value(cls, v, values):
        """Validate measurement value against expected range."""
        measurement_range = values.get('measurement_range')
        if measurement_range:
            if v < measurement_range.min_value or v > measurement_range.max_value:
                # This is a warning, not an error
                pass  # Could add warning logic here
        return v
    
    @validator('signal_quality')
    def validate_signal_quality(cls, v, values):
        """Validate signal quality consistency."""
        noise_level = values.get('noise_level')
        if v == SignalQuality.EXCELLENT and noise_level and noise_level > 0.1:
            # High noise level with excellent signal quality is inconsistent
            pass  # Could add warning logic here
        return v
    
    @root_validator
    def validate_anomaly_consistency(cls, values):
        """Validate anomaly detection consistency."""
        anomaly_detected = values.get('anomaly_detected')
        anomaly_type = values.get('anomaly_type')
        anomaly_severity = values.get('anomaly_severity')
        
        if anomaly_detected:
            if not anomaly_type:
                raise ValueError("anomaly_type must be specified when anomaly_detected is True")
            if not anomaly_severity:
                raise ValueError("anomaly_severity must be specified when anomaly_detected is True")
        else:
            if anomaly_type or anomaly_severity:
                # Clear anomaly fields if no anomaly detected
                values['anomaly_type'] = None
                values['anomaly_severity'] = None
        
        return values
    
    def get_primary_key(self) -> str:
        """Get the primary key field name."""
        return "monitoring_id"
    
    def get_primary_key_value(self) -> Any:
        """Get the primary key value."""
        return self.monitoring_id
    
    def is_measurement_in_range(self) -> bool:
        """
        Check if the measurement value is within the expected range.
        
        Returns:
            True if measurement is in range, False otherwise
        """
        if not self.measurement_range:
            return True  # No range specified, assume valid
        
        return (self.measurement_range.min_value <= self.measurement_value <= 
                self.measurement_range.max_value)
    
    def get_measurement_deviation(self) -> float:
        """
        Calculate deviation from expected range.
        
        Returns:
            Deviation percentage (0-100)
        """
        if not self.measurement_range:
            return 0.0
        
        range_center = (self.measurement_range.min_value + self.measurement_range.max_value) / 2
        range_width = self.measurement_range.max_value - self.measurement_range.min_value
        
        if range_width == 0:
            return 0.0
        
        deviation = abs(self.measurement_value - range_center)
        return (deviation / range_width) * 100
    
    def get_signal_quality_score(self) -> float:
        """
        Get numeric signal quality score.
        
        Returns:
            Signal quality score (0-100)
        """
        quality_scores = {
            SignalQuality.EXCELLENT: 100,
            SignalQuality.GOOD: 80,
            SignalQuality.FAIR: 60,
            SignalQuality.POOR: 40,
            SignalQuality.UNKNOWN: 0
        }
        
        return quality_scores.get(self.signal_quality, 0)
    
    def get_anomaly_risk_score(self) -> float:
        """
        Calculate anomaly risk score based on various factors.
        
        Returns:
            Risk score (0-100)
        """
        risk_score = 0.0
        
        # Base risk from anomaly detection
        if self.anomaly_detected:
            severity_scores = {
                AnomalySeverity.LOW: 20,
                AnomalySeverity.MEDIUM: 40,
                AnomalySeverity.HIGH: 70,
                AnomalySeverity.CRITICAL: 100
            }
            risk_score += severity_scores.get(self.anomaly_severity, 0)
        
        # Risk from signal quality
        signal_quality_score = self.get_signal_quality_score()
        if signal_quality_score < 60:
            risk_score += (100 - signal_quality_score) * 0.3
        
        # Risk from measurement deviation
        deviation = self.get_measurement_deviation()
        if deviation > 10:
            risk_score += min(30, deviation * 0.5)
        
        # Risk from noise level
        if self.noise_level and self.noise_level > 0.1:
            risk_score += min(20, self.noise_level * 100)
        
        # Risk from calibration status
        if self.calibration_status is False:
            risk_score += 15
        
        return min(100, risk_score)
    
    def is_critical_anomaly(self) -> bool:
        """
        Check if this represents a critical anomaly.
        
        Returns:
            True if critical anomaly detected
        """
        return (self.anomaly_detected and 
                self.anomaly_severity == AnomalySeverity.CRITICAL)
    
    def get_measurement_confidence(self) -> float:
        """
        Calculate measurement confidence based on various factors.
        
        Returns:
            Confidence score (0-1)
        """
        confidence = 1.0
        
        # Reduce confidence based on signal quality
        signal_quality_score = self.get_signal_quality_score()
        confidence *= (signal_quality_score / 100)
        
        # Reduce confidence based on noise level
        if self.noise_level:
            noise_penalty = min(0.5, self.noise_level * 5)
            confidence *= (1 - noise_penalty)
        
        # Reduce confidence if not calibrated
        if self.calibration_status is False:
            confidence *= 0.8
        
        # Reduce confidence if measurement is out of range
        if not self.is_measurement_in_range():
            confidence *= 0.7
        
        return max(0.0, confidence)
    
    def get_recommended_action(self) -> str:
        """
        Get recommended action based on the monitoring data.
        
        Returns:
            Recommended action string
        """
        if self.is_critical_anomaly():
            return "IMMEDIATE_ATTENTION_REQUIRED"
        
        risk_score = self.get_anomaly_risk_score()
        if risk_score > 70:
            return "INVESTIGATE_ANOMALY"
        elif risk_score > 40:
            return "MONITOR_CLOSELY"
        elif risk_score > 20:
            return "CONTINUE_MONITORING"
        else:
            return "NORMAL_OPERATION"
    
    def validate_sensor_data(self) -> Dict[str, Any]:
        """
        Validate sensor data quality and consistency.
        
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check measurement range
        if not self.is_measurement_in_range():
            validation_results['warnings'].append(
                f"Measurement value {self.measurement_value} is outside expected range"
            )
        
        # Check signal quality consistency
        if self.signal_quality == SignalQuality.EXCELLENT and self.noise_level and self.noise_level > 0.1:
            validation_results['warnings'].append(
                "Signal quality marked as excellent but noise level is high"
            )
        
        # Check calibration status
        if self.calibration_status is False:
            validation_results['warnings'].append("Sensor is not calibrated")
        
        # Check anomaly consistency
        if self.anomaly_detected and not self.anomaly_type:
            validation_results['errors'].append("Anomaly detected but type not specified")
        
        if validation_results['warnings'] or validation_results['errors']:
            validation_results['valid'] = False
        
        return validation_results
    
    def _calculate_consistency(self) -> float:
        """Calculate data consistency score (0-1)."""
        validation = self.validate_sensor_data()
        if validation['valid']:
            return 1.0
        else:
            # Reduce score based on number of issues
            issue_penalty = (len(validation['warnings']) * 0.1 + 
                           len(validation['errors']) * 0.2)
            return max(0.0, 1.0 - issue_penalty)
    
    def _calculate_accuracy(self) -> float:
        """Calculate data accuracy score (0-1)."""
        # Base accuracy on measurement confidence
        return self.get_measurement_confidence()
    
    def _calculate_validity(self) -> float:
        """Calculate data validity score (0-1)."""
        # Check if all required fields are present and valid
        required_fields = ['monitoring_id', 'process_id', 'sensor_id', 'timestamp', 
                          'sensor_type', 'measurement_value', 'unit']
        valid_fields = 0
        
        for field in required_fields:
            value = getattr(self, field)
            if value is not None:
                valid_fields += 1
        
        return valid_fields / len(required_fields)
