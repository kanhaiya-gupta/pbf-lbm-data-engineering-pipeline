"""
Monitoring type enumeration for PBF-LB/M monitoring systems.
"""

from enum import Enum


class MonitoringType(Enum):
    """
    Enumeration for different types of monitoring in PBF-LB/M processes.
    
    This enum categorizes various monitoring systems and sensors
    used in powder bed fusion laser beam/metal additive manufacturing.
    """
    
    # Process monitoring
    LASER_POWER = "laser_power"
    LASER_SPEED = "laser_speed"
    LASER_FOCUS = "laser_focus"
    SCAN_PATTERN = "scan_pattern"
    LAYER_HEIGHT = "layer_height"
    BUILD_PLATE_TEMPERATURE = "build_plate_temperature"
    
    # Environmental monitoring
    CHAMBER_TEMPERATURE = "chamber_temperature"
    CHAMBER_PRESSURE = "chamber_pressure"
    ATMOSPHERE_COMPOSITION = "atmosphere_composition"
    OXYGEN_LEVEL = "oxygen_level"
    HUMIDITY = "humidity"
    
    # Powder monitoring
    POWDER_TEMPERATURE = "powder_temperature"
    POWDER_DISTRIBUTION = "powder_distribution"
    POWDER_BED_HEIGHT = "powder_bed_height"
    POWDER_QUALITY = "powder_quality"
    POWDER_CONSUMPTION = "powder_consumption"
    
    # Melt pool monitoring
    MELT_POOL_TEMPERATURE = "melt_pool_temperature"
    MELT_POOL_SIZE = "melt_pool_size"
    MELT_POOL_GEOMETRY = "melt_pool_geometry"
    MELT_POOL_DYNAMICS = "melt_pool_dynamics"
    
    # In-situ process monitoring (ISPM)
    ISPM_THERMAL = "ispm_thermal"
    ISPM_OPTICAL = "ispm_optical"
    ISPM_ACOUSTIC = "ispm_acoustic"
    ISPM_ELECTROMAGNETIC = "ispm_electromagnetic"
    ISPM_STRUCTURAL = "ispm_structural"
    
    # Quality monitoring
    DIMENSIONAL_ACCURACY = "dimensional_accuracy"
    SURFACE_QUALITY = "surface_quality"
    DENSITY_MEASUREMENT = "density_measurement"
    DEFECT_DETECTION = "defect_detection"
    MICROSTRUCTURE_ANALYSIS = "microstructure_analysis"
    
    # System monitoring
    EQUIPMENT_STATUS = "equipment_status"
    MAINTENANCE_ALERTS = "maintenance_alerts"
    PERFORMANCE_METRICS = "performance_metrics"
    ENERGY_CONSUMPTION = "energy_consumption"
    CYCLE_TIME = "cycle_time"
    
    # Safety monitoring
    SAFETY_SYSTEMS = "safety_systems"
    EMERGENCY_STOPS = "emergency_stops"
    HAZARD_DETECTION = "hazard_detection"
    PERSONNEL_SAFETY = "personnel_safety"
    
    @classmethod
    def get_process_monitoring(cls):
        """Get process-related monitoring types."""
        return [
            cls.LASER_POWER,
            cls.LASER_SPEED,
            cls.LASER_FOCUS,
            cls.SCAN_PATTERN,
            cls.LAYER_HEIGHT,
            cls.BUILD_PLATE_TEMPERATURE
        ]
    
    @classmethod
    def get_environmental_monitoring(cls):
        """Get environmental monitoring types."""
        return [
            cls.CHAMBER_TEMPERATURE,
            cls.CHAMBER_PRESSURE,
            cls.ATMOSPHERE_COMPOSITION,
            cls.OXYGEN_LEVEL,
            cls.HUMIDITY
        ]
    
    @classmethod
    def get_powder_monitoring(cls):
        """Get powder-related monitoring types."""
        return [
            cls.POWDER_TEMPERATURE,
            cls.POWDER_DISTRIBUTION,
            cls.POWDER_BED_HEIGHT,
            cls.POWDER_QUALITY,
            cls.POWDER_CONSUMPTION
        ]
    
    @classmethod
    def get_melt_pool_monitoring(cls):
        """Get melt pool monitoring types."""
        return [
            cls.MELT_POOL_TEMPERATURE,
            cls.MELT_POOL_SIZE,
            cls.MELT_POOL_GEOMETRY,
            cls.MELT_POOL_DYNAMICS
        ]
    
    @classmethod
    def get_ispm_monitoring(cls):
        """Get in-situ process monitoring types."""
        return [
            cls.ISPM_THERMAL,
            cls.ISPM_OPTICAL,
            cls.ISPM_ACOUSTIC,
            cls.ISPM_ELECTROMAGNETIC,
            cls.ISPM_STRUCTURAL
        ]
    
    @classmethod
    def get_quality_monitoring(cls):
        """Get quality-related monitoring types."""
        return [
            cls.DIMENSIONAL_ACCURACY,
            cls.SURFACE_QUALITY,
            cls.DENSITY_MEASUREMENT,
            cls.DEFECT_DETECTION,
            cls.MICROSTRUCTURE_ANALYSIS
        ]
    
    @classmethod
    def get_system_monitoring(cls):
        """Get system monitoring types."""
        return [
            cls.EQUIPMENT_STATUS,
            cls.MAINTENANCE_ALERTS,
            cls.PERFORMANCE_METRICS,
            cls.ENERGY_CONSUMPTION,
            cls.CYCLE_TIME
        ]
    
    @classmethod
    def get_safety_monitoring(cls):
        """Get safety monitoring types."""
        return [
            cls.SAFETY_SYSTEMS,
            cls.EMERGENCY_STOPS,
            cls.HAZARD_DETECTION,
            cls.PERSONNEL_SAFETY
        ]
    
    def get_frequency_requirement(self):
        """Get the required monitoring frequency for this type."""
        frequency_map = {
            # High frequency (real-time)
            cls.LASER_POWER: "real_time",
            cls.LASER_SPEED: "real_time",
            cls.MELT_POOL_TEMPERATURE: "real_time",
            cls.MELT_POOL_SIZE: "real_time",
            cls.ISPM_THERMAL: "real_time",
            cls.ISPM_OPTICAL: "real_time",
            cls.SAFETY_SYSTEMS: "real_time",
            cls.EMERGENCY_STOPS: "real_time",
            
            # Medium frequency (continuous)
            cls.CHAMBER_TEMPERATURE: "continuous",
            cls.CHAMBER_PRESSURE: "continuous",
            cls.POWDER_TEMPERATURE: "continuous",
            cls.BUILD_PLATE_TEMPERATURE: "continuous",
            cls.ISPM_ACOUSTIC: "continuous",
            cls.ISPM_ELECTROMAGNETIC: "continuous",
            
            # Low frequency (periodic)
            cls.POWDER_QUALITY: "periodic",
            cls.DIMENSIONAL_ACCURACY: "periodic",
            cls.SURFACE_QUALITY: "periodic",
            cls.DENSITY_MEASUREMENT: "periodic",
            cls.MICROSTRUCTURE_ANALYSIS: "periodic",
            cls.EQUIPMENT_STATUS: "periodic",
            cls.MAINTENANCE_ALERTS: "periodic",
            
            # Event-based
            cls.DEFECT_DETECTION: "event_based",
            cls.HAZARD_DETECTION: "event_based",
            cls.PERSONNEL_SAFETY: "event_based",
        }
        return frequency_map.get(self, "continuous")
    
    def get_criticality_level(self):
        """Get the criticality level for this monitoring type."""
        criticality_map = {
            # Critical (safety and process control)
            cls.SAFETY_SYSTEMS: 4,
            cls.EMERGENCY_STOPS: 4,
            cls.LASER_POWER: 4,
            cls.LASER_SPEED: 4,
            cls.HAZARD_DETECTION: 4,
            cls.PERSONNEL_SAFETY: 4,
            
            # High (process quality)
            cls.MELT_POOL_TEMPERATURE: 3,
            cls.MELT_POOL_SIZE: 3,
            cls.ISPM_THERMAL: 3,
            cls.ISPM_OPTICAL: 3,
            cls.CHAMBER_TEMPERATURE: 3,
            cls.CHAMBER_PRESSURE: 3,
            
            # Medium (process optimization)
            cls.POWDER_TEMPERATURE: 2,
            cls.POWDER_DISTRIBUTION: 2,
            cls.BUILD_PLATE_TEMPERATURE: 2,
            cls.ISPM_ACOUSTIC: 2,
            cls.ISPM_ELECTROMAGNETIC: 2,
            
            # Low (analytics and reporting)
            cls.POWDER_QUALITY: 1,
            cls.DIMENSIONAL_ACCURACY: 1,
            cls.SURFACE_QUALITY: 1,
            cls.DENSITY_MEASUREMENT: 1,
            cls.MICROSTRUCTURE_ANALYSIS: 1,
            cls.EQUIPMENT_STATUS: 1,
            cls.MAINTENANCE_ALERTS: 1,
            cls.PERFORMANCE_METRICS: 1,
            cls.ENERGY_CONSUMPTION: 1,
            cls.CYCLE_TIME: 1,
        }
        return criticality_map.get(self, 2)
    
    def is_critical(self):
        """Check if this monitoring type is critical."""
        return self.get_criticality_level() >= 4
    
    def is_high_priority(self):
        """Check if this monitoring type is high priority."""
        return self.get_criticality_level() >= 3
    
    def is_real_time(self):
        """Check if this monitoring type requires real-time monitoring."""
        return self.get_frequency_requirement() == "real_time"
    
    def is_continuous(self):
        """Check if this monitoring type requires continuous monitoring."""
        return self.get_frequency_requirement() == "continuous"
    
    def is_periodic(self):
        """Check if this monitoring type requires periodic monitoring."""
        return self.get_frequency_requirement() == "periodic"
    
    def is_event_based(self):
        """Check if this monitoring type is event-based."""
        return self.get_frequency_requirement() == "event_based"