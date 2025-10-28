// =============================================================================
// PBF-LB/M Manufacturing Knowledge Graph Schema
// Comprehensive schema for Powder Bed Fusion - Laser Beam Melting processes
// =============================================================================

// =============================================================================
// NODE LABELS AND CONSTRAINTS
// =============================================================================

// Core Manufacturing Nodes
CREATE CONSTRAINT process_id_unique IF NOT EXISTS FOR (p:Process) REQUIRE p.process_id IS UNIQUE;
CREATE CONSTRAINT process_timestamp_exists IF NOT EXISTS FOR (p:Process) REQUIRE p.timestamp IS NOT NULL;

CREATE CONSTRAINT machine_id_unique IF NOT EXISTS FOR (m:Machine) REQUIRE m.machine_id IS UNIQUE;
CREATE CONSTRAINT machine_status_exists IF NOT EXISTS FOR (m:Machine) REQUIRE m.status IS NOT NULL;

CREATE CONSTRAINT part_id_unique IF NOT EXISTS FOR (p:Part) REQUIRE p.part_id IS UNIQUE;
CREATE CONSTRAINT build_id_unique IF NOT EXISTS FOR (b:Build) REQUIRE b.build_id IS UNIQUE;

// Material and Quality Nodes
CREATE CONSTRAINT material_type_unique IF NOT EXISTS FOR (m:Material) REQUIRE m.material_type IS UNIQUE;
CREATE CONSTRAINT batch_id_unique IF NOT EXISTS FOR (b:Batch) REQUIRE b.batch_id IS UNIQUE;
CREATE CONSTRAINT quality_grade_exists IF NOT EXISTS FOR (q:Quality) REQUIRE q.grade IS NOT NULL;

// Sensor and Monitoring Nodes
CREATE CONSTRAINT sensor_id_unique IF NOT EXISTS FOR (s:Sensor) REQUIRE s.sensor_id IS UNIQUE;
CREATE CONSTRAINT alert_id_unique IF NOT EXISTS FOR (a:Alert) REQUIRE a.alert_id IS UNIQUE;
CREATE CONSTRAINT measurement_id_unique IF NOT EXISTS FOR (m:Measurement) REQUIRE m.measurement_id IS UNIQUE;

// User and Operator Nodes
CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE;
CREATE CONSTRAINT operator_id_unique IF NOT EXISTS FOR (o:Operator) REQUIRE o.operator_id IS UNIQUE;

// File and Documentation Nodes
CREATE CONSTRAINT image_id_unique IF NOT EXISTS FOR (i:Image) REQUIRE i.image_id IS UNIQUE;
CREATE CONSTRAINT log_id_unique IF NOT EXISTS FOR (l:Log) REQUIRE l.log_id IS UNIQUE;
CREATE CONSTRAINT build_file_id_unique IF NOT EXISTS FOR (bf:BuildFile) REQUIRE bf.build_file_id IS UNIQUE;

// Defect and Quality Control Nodes
CREATE CONSTRAINT defect_id_unique IF NOT EXISTS FOR (d:Defect) REQUIRE d.defect_id IS UNIQUE;
CREATE CONSTRAINT inspection_id_unique IF NOT EXISTS FOR (i:Inspection) REQUIRE i.inspection_id IS UNIQUE;

// =============================================================================
// NODE PROPERTIES AND INDEXES
// =============================================================================

// Process node indexes
CREATE INDEX process_timestamp_index IF NOT EXISTS FOR (p:Process) ON (p.timestamp);
CREATE INDEX process_material_type_index IF NOT EXISTS FOR (p:Process) ON (p.material_type);
CREATE INDEX process_quality_grade_index IF NOT EXISTS FOR (p:Process) ON (p.quality_grade);
CREATE INDEX process_status_index IF NOT EXISTS FOR (p:Process) ON (p.status);
CREATE INDEX process_laser_power_index IF NOT EXISTS FOR (p:Process) ON (p.laser_power);
CREATE INDEX process_scan_speed_index IF NOT EXISTS FOR (p:Process) ON (p.scan_speed);

// Machine node indexes
CREATE INDEX machine_type_index IF NOT EXISTS FOR (m:Machine) ON (m.machine_type);
CREATE INDEX machine_status_index IF NOT EXISTS FOR (m:Machine) ON (m.status);
CREATE INDEX machine_location_index IF NOT EXISTS FOR (m:Machine) ON (m.location);
CREATE INDEX machine_model_index IF NOT EXISTS FOR (m:Machine) ON (m.model);

// Part node indexes
CREATE INDEX part_type_index IF NOT EXISTS FOR (p:Part) ON (p.part_type);
CREATE INDEX part_status_index IF NOT EXISTS FOR (p:Part) ON (p.status);
CREATE INDEX part_material_index IF NOT EXISTS FOR (p:Part) ON (p.material_type);

// Build node indexes
CREATE INDEX build_name_index IF NOT EXISTS FOR (b:Build) ON (b.build_name);
CREATE INDEX build_status_index IF NOT EXISTS FOR (b:Build) ON (b.status);
CREATE INDEX build_created_date_index IF NOT EXISTS FOR (b:Build) ON (b.created_date);

// Material node indexes
CREATE INDEX material_properties_index IF NOT EXISTS FOR (m:Material) ON (m.properties);
CREATE INDEX material_supplier_index IF NOT EXISTS FOR (m:Material) ON (m.supplier);
CREATE INDEX batch_condition_index IF NOT EXISTS FOR (b:Batch) ON (b.condition);

// Quality node indexes
CREATE INDEX quality_metrics_index IF NOT EXISTS FOR (q:Quality) ON (q.metrics);
CREATE INDEX quality_grade_index IF NOT EXISTS FOR (q:Quality) ON (q.grade);
CREATE INDEX quality_standards_index IF NOT EXISTS FOR (q:Quality) ON (q.standards);

// Sensor node indexes
CREATE INDEX sensor_type_index IF NOT EXISTS FOR (s:Sensor) ON (s.sensor_type);
CREATE INDEX sensor_location_index IF NOT EXISTS FOR (s:Sensor) ON (s.location);
CREATE INDEX sensor_model_index IF NOT EXISTS FOR (s:Sensor) ON (s.model);

// Alert node indexes
CREATE INDEX alert_severity_index IF NOT EXISTS FOR (a:Alert) ON (a.severity);
CREATE INDEX alert_status_index IF NOT EXISTS FOR (a:Alert) ON (a.status);
CREATE INDEX alert_timestamp_index IF NOT EXISTS FOR (a:Alert) ON (a.timestamp);

// User node indexes
CREATE INDEX user_role_index IF NOT EXISTS FOR (u:User) ON (u.role);
CREATE INDEX user_department_index IF NOT EXISTS FOR (u:User) ON (u.department);
CREATE INDEX operator_certification_index IF NOT EXISTS FOR (o:Operator) ON (o.certification);

// Image node indexes
CREATE INDEX image_type_index IF NOT EXISTS FOR (i:Image) ON (i.image_type);
CREATE INDEX image_format_index IF NOT EXISTS FOR (i:Image) ON (i.format);
CREATE INDEX image_timestamp_index IF NOT EXISTS FOR (i:Image) ON (i.timestamp);

// Log node indexes
CREATE INDEX log_level_index IF NOT EXISTS FOR (l:Log) ON (l.level);
CREATE INDEX log_source_index IF NOT EXISTS FOR (l:Log) ON (l.source);
CREATE INDEX log_timestamp_index IF NOT EXISTS FOR (l:Log) ON (l.timestamp);

// Defect node indexes
CREATE INDEX defect_type_index IF NOT EXISTS FOR (d:Defect) ON (d.defect_type);
CREATE INDEX defect_severity_index IF NOT EXISTS FOR (d:Defect) ON (d.severity);
CREATE INDEX defect_location_index IF NOT EXISTS FOR (d:Defect) ON (d.location);

// =============================================================================
// RELATIONSHIP TYPES AND PROPERTIES
// =============================================================================

// Core Manufacturing Relationships
// (Process)-[:USES_MACHINE]->(Machine) - Process executed on machine
// (Process)-[:CREATES_PART]->(Part) - Process creates part
// (Process)-[:PART_OF_BUILD]->(Build) - Process belongs to build
// (Process)-[:USES_MATERIAL]->(Material) - Process uses material
// (Process)-[:HAS_QUALITY]->(Quality) - Process has quality metrics
// (Process)-[:MONITORED_BY]->(Sensor) - Process monitored by sensor
// (Process)-[:OPERATED_BY]->(Operator) - Process operated by operator
// (Process)-[:HAS_DEFECT]->(Defect) - Process has defects
// (Process)-[:GENERATES_ALERT]->(Alert) - Process generates alerts
// (Process)-[:CAPTURED_BY]->(Image) - Process captured in images
// (Process)-[:LOGGED_IN]->(Log) - Process logged in logs

// Machine Relationships
// (Machine)-[:HOSTS_PROCESS]->(Process) - Machine hosts process
// (Machine)-[:HAS_SENSOR]->(Sensor) - Machine has sensors
// (Machine)-[:OPERATED_BY]->(Operator) - Machine operated by operator
// (Machine)-[:LOCATED_AT]->(Location) - Machine location
// (Machine)-[:HAS_CAPABILITY]->(Capability) - Machine capabilities

// Part Relationships
// (Part)-[:BELONGS_TO_BUILD]->(Build) - Part belongs to build
// (Part)-[:CREATED_BY_PROCESS]->(Process) - Part created by process
// (Part)-[:MADE_OF_MATERIAL]->(Material) - Part made of material
// (Part)-[:HAS_QUALITY]->(Quality) - Part has quality
// (Part)-[:HAS_DEFECT]->(Defect) - Part has defects
// (Part)-[:INSPECTED_BY]->(Inspection) - Part inspected

// Build Relationships
// (Build)-[:CONTAINS_PROCESS]->(Process) - Build contains processes
// (Build)-[:CONTAINS_PART]->(Part) - Build contains parts
// (Build)-[:USES_MATERIAL]->(Material) - Build uses materials
// (Build)-[:HAS_BUILD_FILE]->(BuildFile) - Build has files
// (Build)-[:CREATED_BY]->(User) - Build created by user

// Material Relationships
// (Material)-[:CONTAINS_BATCH]->(Batch) - Material contains batches
// (Batch)-[:USED_IN_PROCESS]->(Process) - Batch used in process
// (Material)-[:SUPPLIED_BY]->(Supplier) - Material supplied by supplier

// Quality Relationships
// (Quality)-[:MEASURED_BY]->(Sensor) - Quality measured by sensor
// (Quality)-[:INFLUENCES_PROCESS]->(Process) - Quality influences process
// (Quality)-[:VALIDATES_PART]->(Part) - Quality validates part

// Sensor Relationships
// (Sensor)-[:MONITORS_PROCESS]->(Process) - Sensor monitors process
// (Sensor)-[:ATTACHED_TO_MACHINE]->(Machine) - Sensor attached to machine
// (Sensor)-[:GENERATES_MEASUREMENT]->(Measurement) - Sensor generates measurements
// (Sensor)-[:TRIGGERS_ALERT]->(Alert) - Sensor triggers alert

// Alert Relationships
// (Alert)-[:TRIGGERED_BY_SENSOR]->(Sensor) - Alert triggered by sensor
// (Alert)-[:AFFECTS_PROCESS]->(Process) - Alert affects process
// (Alert)-[:NOTIFIES_USER]->(User) - Alert notifies user
// (Alert)-[:RESOLVED_BY]->(User) - Alert resolved by user

// User Relationships
// (User)-[:OPERATES_MACHINE]->(Machine) - User operates machine
// (User)-[:MANAGES_PROCESS]->(Process) - User manages process
// (User)-[:CREATES_BUILD]->(Build) - User creates build
// (User)-[:RESOLVES_ALERT]->(Alert) - User resolves alert

// File Relationships
// (Image)-[:CAPTURES_PROCESS]->(Process) - Image captures process
// (Image)-[:SHOWS_PART]->(Part) - Image shows part
// (Log)-[:RECORDS_PROCESS]->(Process) - Log records process
// (BuildFile)-[:DEFINES_BUILD]->(Build) - BuildFile defines build

// Defect Relationships
// (Defect)-[:DETECTED_BY_SENSOR]->(Sensor) - Defect detected by sensor
// (Defect)-[:AFFECTS_PART]->(Part) - Defect affects part
// (Defect)-[:CAUSED_BY_PROCESS]->(Process) - Defect caused by process
// (Defect)-[:INSPECTED_BY]->(Inspection) - Defect inspected

// =============================================================================
// SAMPLE DATA CREATION QUERIES
// =============================================================================

// Create comprehensive sample data
CREATE (p:Process {
    process_id: 'PROC_001',
    timestamp: datetime('2024-01-15T10:30:00'),
    material_type: 'Ti6Al4V',
    quality_grade: 'A',
    laser_power: 200.0,
    scan_speed: 1000.0,
    layer_thickness: 0.03,
    density: 0.98,
    surface_roughness: 5.2,
    status: 'completed',
    duration: 3600,
    energy_consumption: 150.5,
    powder_usage: 2.3,
    build_temperature: 80.0,
    chamber_pressure: 0.1
});

CREATE (machine:Machine {
    machine_id: 'MACHINE_001',
    machine_type: 'PBF-LB/M',
    model: 'EOS M290',
    status: 'operational',
    location: 'Building A, Floor 2',
    installation_date: date('2023-01-15'),
    max_build_volume: {x: 250, y: 250, z: 325},
    laser_power_max: 400.0,
    layer_thickness_range: {min: 0.02, max: 0.1},
    accuracy: 0.05
});

CREATE (part:Part {
    part_id: 'PART_001',
    part_type: 'turbine_blade',
    material_type: 'Ti6Al4V',
    dimensions: {x: 120.5, y: 45.2, z: 15.8},
    volume: 0.85,
    surface_area: 1250.3,
    weight: 3.2,
    status: 'completed',
    quality_grade: 'A',
    tolerance: 0.05
});

CREATE (build:Build {
    build_id: 'BUILD_001',
    build_name: 'Turbine Assembly',
    status: 'completed',
    created_date: date('2024-01-15'),
    completed_date: date('2024-01-16'),
    total_parts: 5,
    success_rate: 0.95,
    total_duration: 18000,
    material_usage: 15.2
});

CREATE (material:Material {
    material_type: 'Ti6Al4V',
    properties: {
        density: 4.43,
        melting_point: 1668,
        thermal_conductivity: 7.0,
        yield_strength: 880,
        tensile_strength: 950
    },
    supplier: 'MaterialCorp',
    certification: 'AMS4911',
    batch_number: 'BATCH_001',
    condition: 'new',
    storage_temperature: 20.0,
    humidity: 45.0
});

CREATE (quality:Quality {
    grade: 'A',
    metrics: {
        density: 0.98,
        surface_roughness: 5.2,
        dimensional_accuracy: 25.0,
        tensile_strength: 920,
        yield_strength: 850
    },
    standards: ['ISO 2768', 'ASTM F2924'],
    inspector: 'John Doe',
    inspection_date: date('2024-01-16'),
    test_method: 'CT_scan',
    confidence_level: 0.95
});

CREATE (sensor:Sensor {
    sensor_id: 'TEMP_001',
    sensor_type: 'temperature',
    location: 'build_chamber',
    model: 'PT100',
    calibration_date: date('2024-01-01'),
    accuracy: 0.1,
    range: {min: -50, max: 200},
    sampling_rate: 1.0,
    status: 'active'
});

CREATE (alert:Alert {
    alert_id: 'ALERT_001',
    severity: 'warning',
    status: 'active',
    timestamp: datetime('2024-01-15T11:30:00'),
    message: 'Temperature deviation detected',
    threshold: 85.0,
    actual_value: 87.5,
    resolution_time: null
});

CREATE (user:User {
    user_id: 'USER_001',
    username: 'jane.smith',
    name: 'Jane Smith',
    role: 'operator',
    department: 'manufacturing',
    email: 'jane.smith@company.com',
    phone: '+1-555-0123',
    active: true,
    last_login: datetime('2024-01-15T08:00:00')
});

CREATE (operator:Operator {
    operator_id: 'OP_001',
    name: 'Jane Smith',
    certification: 'PBF-LB/M Level 2',
    experience_years: 5,
    shift: 'day',
    machine_authorization: ['MACHINE_001', 'MACHINE_002'],
    training_completed: ['safety', 'quality_control', 'maintenance']
});

CREATE (image:Image {
    image_id: 'IMG_001',
    image_type: 'process_monitoring',
    format: 'PNG',
    resolution: {width: 1920, height: 1080},
    file_size: 2048576,
    timestamp: datetime('2024-01-15T10:45:00'),
    camera_position: {x: 10.5, y: 20.3, z: 2.1},
    lighting_conditions: 'standard',
    quality_score: 0.92
});

CREATE (log:Log {
    log_id: 'LOG_001',
    level: 'INFO',
    source: 'process_monitor',
    message: 'Process started successfully',
    timestamp: datetime('2024-01-15T10:30:00'),
    component: 'laser_control',
    session_id: 'SESSION_001',
    user_id: 'USER_001'
});

CREATE (defect:Defect {
    defect_id: 'DEF_001',
    defect_type: 'porosity',
    severity: 'minor',
    location: {x: 10.5, y: 20.3, z: 5.2},
    size: 0.1,
    detection_method: 'CT_scan',
    confidence: 0.85,
    status: 'detected',
    timestamp: datetime('2024-01-15T15:00:00')
});

// =============================================================================
// RELATIONSHIP CREATION QUERIES
// =============================================================================

// Create comprehensive relationships
MATCH (p:Process {process_id: 'PROC_001'})
MATCH (machine:Machine {machine_id: 'MACHINE_001'})
MATCH (part:Part {part_id: 'PART_001'})
MATCH (build:Build {build_id: 'BUILD_001'})
MATCH (material:Material {material_type: 'Ti6Al4V'})
MATCH (quality:Quality {grade: 'A'})
MATCH (sensor:Sensor {sensor_id: 'TEMP_001'})
MATCH (alert:Alert {alert_id: 'ALERT_001'})
MATCH (user:User {user_id: 'USER_001'})
MATCH (operator:Operator {operator_id: 'OP_001'})
MATCH (image:Image {image_id: 'IMG_001'})
MATCH (log:Log {log_id: 'LOG_001'})
MATCH (defect:Defect {defect_id: 'DEF_001'})

// Core manufacturing relationships
CREATE (p)-[:USES_MACHINE {duration: 3600, start_time: datetime('2024-01-15T10:30:00')}]->(machine)
CREATE (p)-[:CREATES_PART {quantity: 1, success_rate: 0.95}]->(part)
CREATE (p)-[:PART_OF_BUILD {sequence: 1, priority: 'high'}]->(build)
CREATE (p)-[:USES_MATERIAL {quantity: 2.3, unit: 'kg'}]->(material)
CREATE (p)-[:HAS_QUALITY {measured_at: datetime('2024-01-15T14:30:00')}]->(quality)
CREATE (p)-[:MONITORED_BY {sampling_rate: 1.0, active: true}]->(sensor)
CREATE (p)-[:OPERATED_BY {shift: 'day', experience_level: 'expert'}]->(operator)
CREATE (p)-[:HAS_DEFECT {detected_at: datetime('2024-01-15T15:00:00')}]->(defect)
CREATE (p)-[:GENERATES_ALERT {triggered_at: datetime('2024-01-15T11:30:00')}]->(alert)
CREATE (p)-[:CAPTURED_BY {timestamp: datetime('2024-01-15T10:45:00')}]->(image)
CREATE (p)-[:LOGGED_IN {level: 'INFO', timestamp: datetime('2024-01-15T10:30:00')}]->(log)

// Reverse relationships
CREATE (machine)-[:HOSTS_PROCESS {capacity: 1, utilization: 0.85}]->(p)
CREATE (part)-[:CREATED_BY_PROCESS {creation_time: datetime('2024-01-15T12:30:00')}]->(p)
CREATE (build)-[:CONTAINS_PROCESS {sequence: 1, status: 'completed'}]->(p)
CREATE (material)-[:USED_IN_PROCESS {consumption_rate: 0.64}]->(p)
CREATE (quality)-[:VALIDATES_PROCESS {confidence: 0.95}]->(p)
CREATE (sensor)-[:MONITORS_PROCESS {coverage: 1.0, accuracy: 0.98}]->(p)
CREATE (operator)-[:OPERATES_PROCESS {supervision_level: 'full'}]->(p)
CREATE (defect)-[:AFFECTS_PROCESS {impact: 'minor', resolution: 'none'}]->(p)
CREATE (alert)-[:AFFECTS_PROCESS {severity: 'warning', resolved: false}]->(p)
CREATE (image)-[:CAPTURES_PROCESS {quality: 0.92, purpose: 'monitoring'}]->(p)
CREATE (log)-[:RECORDS_PROCESS {detail_level: 'comprehensive'}]->(p)

// User and operator relationships
CREATE (user)-[:OPERATES_MACHINE {authorization_level: 'full'}]->(machine)
CREATE (user)-[:MANAGES_PROCESS {responsibility: 'primary'}]->(p)
CREATE (user)-[:CREATES_BUILD {design_authority: true}]->(build)
CREATE (user)-[:RESOLVES_ALERT {response_time: 300}]->(alert)

// Machine and sensor relationships
CREATE (machine)-[:HAS_SENSOR {installation_date: date('2023-01-15')}]->(sensor)
CREATE (sensor)-[:ATTACHED_TO_MACHINE {position: 'chamber_center'}]->(machine)
CREATE (sensor)-[:GENERATES_MEASUREMENT {frequency: 1.0}]->(alert)
CREATE (sensor)-[:TRIGGERS_ALERT {threshold_exceeded: true}]->(alert)

// Quality and inspection relationships
CREATE (quality)-[:MEASURED_BY {instrument: 'CT_scanner'}]->(sensor)
CREATE (quality)-[:INFLUENCES_PROCESS {correlation: 0.85}]->(p)
CREATE (quality)-[:VALIDATES_PART {acceptance_criteria: 'met'}]->(part)

// Defect relationships
CREATE (defect)-[:DETECTED_BY_SENSOR {detection_confidence: 0.85}]->(sensor)
CREATE (defect)-[:AFFECTS_PART {severity_impact: 'minor'}]->(part)
CREATE (defect)-[:CAUSED_BY_PROCESS {root_cause: 'parameter_deviation'}]->(p)

// File relationships
CREATE (image)-[:SHOWS_PART {clarity: 0.92, angle: 'top_view'}]->(part)
CREATE (log)-[:RECORDS_PROCESS {detail_level: 'comprehensive'}]->(p)
CREATE (build)-[:HAS_BUILD_FILE {file_type: 'STL', version: '1.0'}]->(build)


// =============================================================================
// NEW NODE TYPES - COMPREHENSIVE MANUFACTURING SCHEMA
// =============================================================================

// =============================================================================
// IMAGE NODE CONSTRAINTS
// =============================================================================

// Thermal Image nodes
CREATE CONSTRAINT thermal_image_id_unique IF NOT EXISTS FOR (ti:ThermalImage) REQUIRE ti.thermal_id IS UNIQUE;
CREATE CONSTRAINT thermal_image_timestamp_exists IF NOT EXISTS FOR (ti:ThermalImage) REQUIRE ti.timestamp IS NOT NULL;
CREATE CONSTRAINT thermal_image_temperature_range_exists IF NOT EXISTS FOR (ti:ThermalImage) REQUIRE ti.temperature_range IS NOT NULL;

// Process Image nodes
CREATE CONSTRAINT process_image_id_unique IF NOT EXISTS FOR (pi:ProcessImage) REQUIRE pi.image_id IS UNIQUE;
CREATE CONSTRAINT process_image_timestamp_exists IF NOT EXISTS FOR (pi:ProcessImage) REQUIRE pi.timestamp IS NOT NULL;

// CT Scan Image nodes
CREATE CONSTRAINT ct_scan_id_unique IF NOT EXISTS FOR (ct:CTScanImage) REQUIRE ct.scan_id IS UNIQUE;
CREATE CONSTRAINT ct_scan_timestamp_exists IF NOT EXISTS FOR (ct:CTScanImage) REQUIRE ct.timestamp IS NOT NULL;

// Powder Bed Image nodes
CREATE CONSTRAINT powder_bed_image_id_unique IF NOT EXISTS FOR (pbi:PowderBedImage) REQUIRE pbi.image_id IS UNIQUE;
CREATE CONSTRAINT powder_bed_image_timestamp_exists IF NOT EXISTS FOR (pbi:PowderBedImage) REQUIRE pbi.timestamp IS NOT NULL;

// =============================================================================
// FILE NODE CONSTRAINTS
// =============================================================================

// Build File nodes
CREATE CONSTRAINT build_file_id_unique IF NOT EXISTS FOR (bf:BuildFile) REQUIRE bf.file_id IS UNIQUE;
CREATE CONSTRAINT build_file_checksum_exists IF NOT EXISTS FOR (bf:BuildFile) REQUIRE bf.checksum IS NOT NULL;

// Model File nodes
CREATE CONSTRAINT model_file_id_unique IF NOT EXISTS FOR (mf:ModelFile) REQUIRE mf.model_id IS UNIQUE;
CREATE CONSTRAINT model_file_dimensions_exists IF NOT EXISTS FOR (mf:ModelFile) REQUIRE mf.dimensions IS NOT NULL;

// Log File nodes
CREATE CONSTRAINT log_file_id_unique IF NOT EXISTS FOR (lf:LogFile) REQUIRE lf.log_id IS UNIQUE;
CREATE CONSTRAINT log_file_start_time_exists IF NOT EXISTS FOR (lf:LogFile) REQUIRE lf.start_time IS NOT NULL;

// =============================================================================
// CACHE NODE CONSTRAINTS
// =============================================================================

// Process Cache nodes
CREATE CONSTRAINT process_cache_id_unique IF NOT EXISTS FOR (pc:ProcessCache) REQUIRE pc.cache_id IS UNIQUE;
CREATE CONSTRAINT process_cache_key_exists IF NOT EXISTS FOR (pc:ProcessCache) REQUIRE pc.key IS NOT NULL;

// Analytics Cache nodes
CREATE CONSTRAINT analytics_cache_id_unique IF NOT EXISTS FOR (ac:AnalyticsCache) REQUIRE ac.cache_id IS UNIQUE;
CREATE CONSTRAINT analytics_cache_key_exists IF NOT EXISTS FOR (ac:AnalyticsCache) REQUIRE ac.cache_key IS NOT NULL;

// =============================================================================
// QUEUE NODE CONSTRAINTS
// =============================================================================

// Job Queue nodes
CREATE CONSTRAINT job_queue_id_unique IF NOT EXISTS FOR (jq:JobQueue) REQUIRE jq.job_id IS UNIQUE;
CREATE CONSTRAINT job_queue_priority_exists IF NOT EXISTS FOR (jq:JobQueue) REQUIRE jq.priority IS NOT NULL;

// =============================================================================
// SESSION NODE CONSTRAINTS
// =============================================================================

// User Session nodes
CREATE CONSTRAINT user_session_id_unique IF NOT EXISTS FOR (us:UserSession) REQUIRE us.session_id IS UNIQUE;
CREATE CONSTRAINT user_session_created_at_exists IF NOT EXISTS FOR (us:UserSession) REQUIRE us.created_at IS NOT NULL;

// =============================================================================
// READING NODE CONSTRAINTS
// =============================================================================

// Sensor Reading nodes
CREATE CONSTRAINT sensor_reading_id_unique IF NOT EXISTS FOR (sr:SensorReading) REQUIRE sr.reading_id IS UNIQUE;
CREATE CONSTRAINT sensor_reading_timestamp_exists IF NOT EXISTS FOR (sr:SensorReading) REQUIRE sr.timestamp IS NOT NULL;

// =============================================================================
// EVENT NODE CONSTRAINTS
// =============================================================================

// Process Monitoring nodes
CREATE CONSTRAINT process_monitoring_id_unique IF NOT EXISTS FOR (pm:ProcessMonitoring) REQUIRE pm.event_id IS UNIQUE;
CREATE CONSTRAINT process_monitoring_timestamp_exists IF NOT EXISTS FOR (pm:ProcessMonitoring) REQUIRE pm.timestamp IS NOT NULL;

// Machine Status nodes
CREATE CONSTRAINT machine_status_id_unique IF NOT EXISTS FOR (ms:MachineStatus) REQUIRE ms.status_id IS UNIQUE;
CREATE CONSTRAINT machine_status_timestamp_exists IF NOT EXISTS FOR (ms:MachineStatus) REQUIRE ms.timestamp IS NOT NULL;

// Alert Event nodes
CREATE CONSTRAINT alert_event_id_unique IF NOT EXISTS FOR (ae:AlertEvent) REQUIRE ae.alert_id IS UNIQUE;
CREATE CONSTRAINT alert_event_timestamp_exists IF NOT EXISTS FOR (ae:AlertEvent) REQUIRE ae.timestamp IS NOT NULL;

// =============================================================================
// INDEXES FOR NEW NODE TYPES
// =============================================================================

// Image indexes
CREATE INDEX thermal_image_timestamp_index IF NOT EXISTS FOR (ti:ThermalImage) ON (ti.timestamp);
CREATE INDEX thermal_image_temperature_index IF NOT EXISTS FOR (ti:ThermalImage) ON (ti.temperature_range);
CREATE INDEX process_image_timestamp_index IF NOT EXISTS FOR (pi:ProcessImage) ON (pi.timestamp);
CREATE INDEX ct_scan_timestamp_index IF NOT EXISTS FOR (ct:CTScanImage) ON (ct.timestamp);
CREATE INDEX powder_bed_layer_index IF NOT EXISTS FOR (pbi:PowderBedImage) ON (pbi.layer_number);

// File indexes
CREATE INDEX build_file_created_index IF NOT EXISTS FOR (bf:BuildFile) ON (bf.created_at);
CREATE INDEX model_file_format_index IF NOT EXISTS FOR (mf:ModelFile) ON (mf.format);
CREATE INDEX log_file_level_index IF NOT EXISTS FOR (lf:LogFile) ON (lf.level);

// Cache indexes
CREATE INDEX process_cache_ttl_index IF NOT EXISTS FOR (pc:ProcessCache) ON (pc.expires_at);
CREATE INDEX analytics_cache_type_index IF NOT EXISTS FOR (ac:AnalyticsCache) ON (ac.analysis_type);

// Queue indexes
CREATE INDEX job_queue_priority_index IF NOT EXISTS FOR (jq:JobQueue) ON (jq.priority);
CREATE INDEX job_queue_status_index IF NOT EXISTS FOR (jq:JobQueue) ON (jq.status);

// Session indexes
CREATE INDEX user_session_status_index IF NOT EXISTS FOR (us:UserSession) ON (us.status);
CREATE INDEX user_session_expires_index IF NOT EXISTS FOR (us:UserSession) ON (us.expires_at);

// Reading indexes
CREATE INDEX sensor_reading_type_index IF NOT EXISTS FOR (sr:SensorReading) ON (sr.reading_type);
CREATE INDEX sensor_reading_quality_index IF NOT EXISTS FOR (sr:SensorReading) ON (sr.quality);

// Event indexes
CREATE INDEX process_monitoring_severity_index IF NOT EXISTS FOR (pm:ProcessMonitoring) ON (pm.severity);
CREATE INDEX machine_status_type_index IF NOT EXISTS FOR (ms:MachineStatus) ON (ms.status_type);
CREATE INDEX alert_event_severity_index IF NOT EXISTS FOR (ae:AlertEvent) ON (ae.severity);

// =============================================================================
// SAMPLE DATA FOR NEW NODE TYPES
// =============================================================================

// Sample Thermal Image
CREATE (ti:ThermalImage {
    thermal_id: 'THERMAL_001',
    process_id: 'PROC_001',
    thermal_type: 'build_monitoring',
    file_path: '/data/thermal/thermal_001.tiff',
    file_size: 2048576,
    dimensions: {width: 640, height: 480},
    format: 'TIFF',
    temperature_range: {min: 25.5, max: 1850.2},
    emissivity: 0.95,
    ambient_temperature: 22.0,
    timestamp: datetime('2024-01-15T10:30:00Z'),
    camera_position: {x: 1.5, y: 2.0, z: 0.8},
    distance_to_target: 0.5,
    field_of_view: {horizontal: 45.0, vertical: 35.0},
    thermal_resolution: 0.1,
    quality_score: 0.92,
    hot_spots_detected: [
        {location: {x: 320, y: 240}, temperature: 1850.2, confidence: 0.95}
    ],
    cold_spots_detected: [
        {location: {x: 100, y: 100}, temperature: 25.5, confidence: 0.88}
    ]
});

// Sample Process Image
CREATE (pi:ProcessImage {
    image_id: 'PROC_IMG_001',
    process_id: 'PROC_001',
    image_type: 'build_progress',
    file_path: '/data/images/proc_001.jpg',
    file_size: 1024000,
    dimensions: {width: 1920, height: 1080},
    format: 'JPEG',
    resolution: 300.0,
    timestamp: datetime('2024-01-15T10:30:00Z')
});

// Sample CT Scan Image
CREATE (ct:CTScanImage {
    scan_id: 'CT_001',
    part_id: 'PART_001',
    scan_type: 'quality_inspection',
    file_path: '/data/ct/ct_001.dcm',
    file_size: 5242880,
    voxel_size: {x: 0.1, y: 0.1, z: 0.1},
    scan_resolution: {x: 512, y: 512, z: 256},
    timestamp: datetime('2024-01-15T11:00:00Z'),
    quality_score: 0.95
});

// Sample Powder Bed Image
CREATE (pbi:PowderBedImage {
    image_id: 'POWDER_001',
    build_id: 'BUILD_001',
    layer_number: 15,
    image_type: 'layer_inspection',
    file_path: '/data/powder/layer_015.jpg',
    file_size: 512000,
    dimensions: {width: 1280, height: 720},
    timestamp: datetime('2024-01-15T10:45:00Z'),
    powder_density: 0.85,
    defects_detected: ['sparse_powder', 'uneven_distribution']
});

// Sample Build File
CREATE (bf:BuildFile {
    file_id: 'BUILD_FILE_001',
    build_id: 'BUILD_001',
    file_type: 'configuration',
    file_path: '/data/builds/build_001.json',
    file_size: 8192,
    format: 'JSON',
    version: '1.2',
    checksum: 'a1b2c3d4e5f6789012345678901234567890abcd',
    created_at: datetime('2024-01-15T09:00:00Z')
});

// Sample Model File
CREATE (mf:ModelFile {
    model_id: 'MODEL_001',
    part_id: 'PART_001',
    file_type: '3d_model',
    file_path: '/data/models/part_001.stl',
    file_size: 1048576,
    format: 'STL',
    version: '2.0',
    dimensions: {x: 100.0, y: 50.0, z: 25.0},
    volume: 125000.0,
    surface_area: 15000.0,
    complexity_score: 0.75,
    created_at: datetime('2024-01-15T08:30:00Z')
});

// Sample Log File
CREATE (lf:LogFile {
    log_id: 'LOG_001',
    process_id: 'PROC_001',
    log_type: 'process_log',
    file_path: '/data/logs/proc_001.log',
    file_size: 256000,
    format: 'TEXT',
    level: 'INFO',
    entries_count: 1250,
    start_time: datetime('2024-01-15T10:00:00Z'),
    end_time: datetime('2024-01-15T12:00:00Z'),
    duration: 7200.0
});

// Sample Process Cache
CREATE (pc:ProcessCache {
    cache_id: 'CACHE_001',
    process_id: 'PROC_001',
    cache_type: 'process_data',
    key: 'proc_001_params',
    value: '{"laser_power": 200, "scan_speed": 1000}',
    size: 64,
    ttl: 3600,
    created_at: datetime('2024-01-15T10:30:00Z'),
    expires_at: datetime('2024-01-15T11:30:00Z'),
    access_count: 5,
    last_accessed: datetime('2024-01-15T10:45:00Z')
});

// Sample Analytics Cache
CREATE (ac:AnalyticsCache {
    cache_id: 'ANALYTICS_001',
    analysis_type: 'quality_prediction',
    cache_key: 'quality_pred_001',
    result_data: '{"predicted_quality": 0.92, "confidence": 0.88}',
    size: 128,
    ttl: 1800,
    created_at: datetime('2024-01-15T10:30:00Z'),
    expires_at: datetime('2024-01-15T11:00:00Z'),
    computation_time: 2.5,
    accuracy: 0.88
});

// Sample Job Queue
CREATE (jq:JobQueue {
    job_id: 'JOB_001',
    queue_name: 'processing_queue',
    job_type: 'data_analysis',
    priority: 3,
    status: 'pending',
    payload: '{"analysis_type": "quality_check", "data_id": "PROC_001"}',
    created_at: datetime('2024-01-15T10:30:00Z'),
    retry_count: 0,
    max_retries: 3,
    timeout: 3600
});

// Sample User Session
CREATE (us:UserSession {
    session_id: 'SESSION_001',
    user_id: 'USER_001',
    session_type: 'web',
    status: 'active',
    created_at: datetime('2024-01-15T10:00:00Z'),
    last_activity: datetime('2024-01-15T10:30:00Z'),
    expires_at: datetime('2024-01-15T18:00:00Z'),
    ip_address: '192.168.1.100',
    user_agent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    permissions: ['read', 'write', 'admin']
});

// Sample Sensor Reading
CREATE (sr:SensorReading {
    reading_id: 'READING_001',
    sensor_id: 'SENSOR_001',
    reading_type: 'temperature',
    value: 1850.5,
    unit: 'Celsius',
    timestamp: datetime('2024-01-15T10:30:00Z'),
    quality: 0.95,
    status: 'valid',
    location: {x: 1.0, y: 2.0, z: 0.5}
});

// Sample Process Monitoring
CREATE (pm:ProcessMonitoring {
    event_id: 'EVENT_001',
    process_id: 'PROC_001',
    event_type: 'parameter_deviation',
    severity: 'warning',
    message: 'Laser power exceeded threshold by 5%',
    timestamp: datetime('2024-01-15T10:30:00Z'),
    parameters: {threshold: 200, actual: 210, deviation: 5.0},
    status: 'active',
    resolved: false
});

// Sample Machine Status
CREATE (ms:MachineStatus {
    status_id: 'STATUS_001',
    machine_id: 'MACHINE_001',
    status_type: 'operational',
    status_value: 'running',
    timestamp: datetime('2024-01-15T10:30:00Z'),
    duration: 3600.0,
    reason: 'Scheduled maintenance completed',
    operator_id: 'OPERATOR_001'
});

// Sample Alert Event
CREATE (ae:AlertEvent {
    alert_id: 'ALERT_001',
    source_id: 'PROC_001',
    alert_type: 'temperature_anomaly',
    severity: 'high',
    message: 'Temperature spike detected in build chamber',
    timestamp: datetime('2024-01-15T10:30:00Z'),
    status: 'active',
    acknowledged: false,
    resolved: false
});

// =============================================================================
// RELATIONSHIPS FOR NEW NODE TYPES
// =============================================================================

// Image relationships
CREATE (ti)-[:CAPTURES_PROCESS {thermal_analysis: true}]->(p)
CREATE (pi)-[:DOCUMENTS_PROCESS {image_quality: 0.92}]->(p)
CREATE (ct)-[:SCANS_PART {scan_quality: 0.95}]->(part)
CREATE (pbi)-[:SHOWS_LAYER {layer_quality: 0.85}]->(build)

// File relationships
CREATE (bf)-[:CONFIGURES_BUILD {file_version: '1.2'}]->(build)
CREATE (mf)-[:MODELS_PART {model_complexity: 0.75}]->(part)
CREATE (lf)-[:LOGS_PROCESS {log_level: 'INFO'}]->(p)

// Cache relationships
CREATE (pc)-[:CACHES_PROCESS {cache_hit_rate: 0.85}]->(p)
CREATE (ac)-[:CACHES_ANALYSIS {analysis_accuracy: 0.88}]->(p)

// Queue relationships
CREATE (jq)-[:QUEUES_FOR_PROCESS {job_priority: 3}]->(p)

// Session relationships
CREATE (us)-[:AUTHENTICATES_USER {session_duration: 28800}]->(user)

// Reading relationships
CREATE (sr)-[:READS_FROM_SENSOR {reading_quality: 0.95}]->(sensor)

// Event relationships
CREATE (pm)-[:MONITORS_PROCESS {event_severity: 'warning'}]->(p)
CREATE (ms)-[:UPDATES_MACHINE {status_change: 'operational'}]->(machine)
CREATE (ae)-[:ALERTS_FOR_PROCESS {alert_severity: 'high'}]->(p)
