// =============================================================================
// Sample Data for PBF-LB/M Manufacturing Knowledge Graph
// Comprehensive sample data for testing and demonstration
// =============================================================================

// =============================================================================
// CLEAR EXISTING DATA
// =============================================================================

// Clear all existing data
MATCH (n) DETACH DELETE n;

// =============================================================================
// SAMPLE MACHINES
// =============================================================================

// Create sample machines
CREATE (m1:Machine {
    machine_id: 'MACHINE_001',
    machine_type: 'PBF-LB/M',
    model: 'EOS M290',
    status: 'operational',
    location: 'Building A, Floor 2',
    installation_date: date('2023-01-15'),
    max_build_volume: {x: 250, y: 250, z: 325},
    laser_power_max: 400.0,
    layer_thickness_range: {min: 0.02, max: 0.1},
    accuracy: 0.05,
    maintenance_date: date('2024-01-01'),
    utilization_rate: 0.85
});

CREATE (m2:Machine {
    machine_id: 'MACHINE_002',
    machine_type: 'PBF-LB/M',
    model: 'EOS M400',
    status: 'operational',
    location: 'Building A, Floor 2',
    installation_date: date('2023-03-20'),
    max_build_volume: {x: 400, y: 400, z: 400},
    laser_power_max: 1000.0,
    layer_thickness_range: {min: 0.02, max: 0.1},
    accuracy: 0.03,
    maintenance_date: date('2024-01-15'),
    utilization_rate: 0.92
});

CREATE (m3:Machine {
    machine_id: 'MACHINE_003',
    machine_type: 'PBF-LB/M',
    model: 'SLM 280',
    status: 'maintenance',
    location: 'Building B, Floor 1',
    installation_date: date('2022-11-10'),
    max_build_volume: {x: 280, y: 280, z: 350},
    laser_power_max: 700.0,
    layer_thickness_range: {min: 0.02, max: 0.1},
    accuracy: 0.04,
    maintenance_date: date('2024-01-20'),
    utilization_rate: 0.78
});

// =============================================================================
// SAMPLE MATERIALS
// =============================================================================

// Create sample materials
CREATE (mat1:Material {
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
    humidity: 45.0,
    shelf_life: 365
});

CREATE (mat2:Material {
    material_type: 'Inconel 718',
    properties: {
        density: 8.19,
        melting_point: 1260,
        thermal_conductivity: 11.4,
        yield_strength: 1035,
        tensile_strength: 1275
    },
    supplier: 'SuperAlloys Inc',
    certification: 'AMS5662',
    batch_number: 'BATCH_002',
    condition: 'new',
    storage_temperature: 25.0,
    humidity: 40.0,
    shelf_life: 730
});

CREATE (mat3:Material {
    material_type: 'Stainless Steel 316L',
    properties: {
        density: 7.9,
        melting_point: 1400,
        thermal_conductivity: 16.0,
        yield_strength: 290,
        tensile_strength: 580
    },
    supplier: 'SteelWorks Ltd',
    certification: 'ASTM A240',
    batch_number: 'BATCH_003',
    condition: 'new',
    storage_temperature: 22.0,
    humidity: 50.0,
    shelf_life: 1095
});

// =============================================================================
// SAMPLE USERS AND OPERATORS
// =============================================================================

// Create sample users
CREATE (u1:User {
    user_id: 'USER_001',
    username: 'jane.smith',
    name: 'Jane Smith',
    role: 'operator',
    department: 'manufacturing',
    email: 'jane.smith@company.com',
    phone: '+1-555-0123',
    active: true,
    last_login: datetime('2024-01-15T08:00:00'),
    hire_date: date('2022-03-15')
});

CREATE (u2:User {
    user_id: 'USER_002',
    username: 'john.doe',
    name: 'John Doe',
    role: 'engineer',
    department: 'quality',
    email: 'john.doe@company.com',
    phone: '+1-555-0124',
    active: true,
    last_login: datetime('2024-01-15T09:30:00'),
    hire_date: date('2021-08-20')
});

CREATE (u3:User {
    user_id: 'USER_003',
    username: 'mike.wilson',
    name: 'Mike Wilson',
    role: 'supervisor',
    department: 'manufacturing',
    email: 'mike.wilson@company.com',
    phone: '+1-555-0125',
    active: true,
    last_login: datetime('2024-01-15T07:45:00'),
    hire_date: date('2020-01-10')
});

// Create sample operators
CREATE (o1:Operator {
    operator_id: 'OP_001',
    name: 'Jane Smith',
    certification: 'PBF-LB/M Level 2',
    experience_years: 5,
    shift: 'day',
    machine_authorization: ['MACHINE_001', 'MACHINE_002'],
    training_completed: ['safety', 'quality_control', 'maintenance'],
    performance_rating: 4.8
});

CREATE (o2:Operator {
    operator_id: 'OP_002',
    name: 'John Doe',
    certification: 'PBF-LB/M Level 3',
    experience_years: 8,
    shift: 'night',
    machine_authorization: ['MACHINE_001', 'MACHINE_002', 'MACHINE_003'],
    training_completed: ['safety', 'quality_control', 'maintenance', 'advanced_techniques'],
    performance_rating: 4.9
});

CREATE (o3:Operator {
    operator_id: 'OP_003',
    name: 'Mike Wilson',
    certification: 'PBF-LB/M Level 1',
    experience_years: 2,
    shift: 'day',
    machine_authorization: ['MACHINE_001'],
    training_completed: ['safety', 'quality_control'],
    performance_rating: 4.2
});

// =============================================================================
// SAMPLE SENSORS
// =============================================================================

// Create sample sensors
CREATE (s1:Sensor {
    sensor_id: 'TEMP_001',
    sensor_type: 'temperature',
    location: 'build_chamber',
    model: 'PT100',
    calibration_date: date('2024-01-01'),
    accuracy: 0.1,
    range: {min: -50, max: 200},
    sampling_rate: 1.0,
    status: 'active',
    last_reading: 85.5
});

CREATE (s2:Sensor {
    sensor_id: 'TEMP_002',
    sensor_type: 'temperature',
    location: 'powder_bed',
    model: 'PT100',
    calibration_date: date('2024-01-01'),
    accuracy: 0.1,
    range: {min: -50, max: 200},
    sampling_rate: 1.0,
    status: 'active',
    last_reading: 82.3
});

CREATE (s3:Sensor {
    sensor_id: 'PRESSURE_001',
    sensor_type: 'pressure',
    location: 'build_chamber',
    model: 'MPX5700',
    calibration_date: date('2024-01-01'),
    accuracy: 0.01,
    range: {min: 0, max: 1},
    sampling_rate: 0.5,
    status: 'active',
    last_reading: 0.1
});

CREATE (s4:Sensor {
    sensor_id: 'LASER_001',
    sensor_type: 'laser_power',
    location: 'laser_head',
    model: 'LPM-100',
    calibration_date: date('2024-01-01'),
    accuracy: 0.5,
    range: {min: 0, max: 1000},
    sampling_rate: 10.0,
    status: 'active',
    last_reading: 200.0
});

// =============================================================================
// SAMPLE BUILDS
// =============================================================================

// Create sample builds
CREATE (b1:Build {
    build_id: 'BUILD_001',
    build_name: 'Turbine Assembly',
    status: 'completed',
    created_date: date('2024-01-15'),
    completed_date: date('2024-01-16'),
    total_parts: 5,
    success_rate: 0.95,
    total_duration: 18000,
    material_usage: 15.2,
    energy_consumption: 2500.5,
    quality_grade: 'A'
});

CREATE (b2:Build {
    build_id: 'BUILD_002',
    build_name: 'Engine Component',
    status: 'in_progress',
    created_date: date('2024-01-20'),
    completed_date: null,
    total_parts: 3,
    success_rate: 0.0,
    total_duration: 0,
    material_usage: 8.5,
    energy_consumption: 0.0,
    quality_grade: null
});

CREATE (b3:Build {
    build_id: 'BUILD_003',
    build_name: 'Prototype Part',
    status: 'completed',
    created_date: date('2024-01-10'),
    completed_date: date('2024-01-12'),
    total_parts: 1,
    success_rate: 1.0,
    total_duration: 7200,
    material_usage: 2.1,
    energy_consumption: 1200.0,
    quality_grade: 'A'
});

// =============================================================================
// SAMPLE PROCESSES
// =============================================================================

// Create sample processes
CREATE (p1:Process {
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
    chamber_pressure: 0.1,
    hatch_spacing: 0.1,
    exposure_time: 0.1
});

CREATE (p2:Process {
    process_id: 'PROC_002',
    timestamp: datetime('2024-01-15T14:00:00'),
    material_type: 'Ti6Al4V',
    quality_grade: 'B',
    laser_power: 180.0,
    scan_speed: 1200.0,
    layer_thickness: 0.03,
    density: 0.95,
    surface_roughness: 6.8,
    status: 'completed',
    duration: 3200,
    energy_consumption: 140.2,
    powder_usage: 2.1,
    build_temperature: 85.0,
    chamber_pressure: 0.1,
    hatch_spacing: 0.12,
    exposure_time: 0.08
});

CREATE (p3:Process {
    process_id: 'PROC_003',
    timestamp: datetime('2024-01-16T09:00:00'),
    material_type: 'Inconel 718',
    quality_grade: 'A',
    laser_power: 300.0,
    scan_speed: 800.0,
    layer_thickness: 0.04,
    density: 0.99,
    surface_roughness: 4.5,
    status: 'completed',
    duration: 4200,
    energy_consumption: 200.8,
    powder_usage: 3.2,
    build_temperature: 90.0,
    chamber_pressure: 0.05,
    hatch_spacing: 0.08,
    exposure_time: 0.12
});

CREATE (p4:Process {
    process_id: 'PROC_004',
    timestamp: datetime('2024-01-16T15:30:00'),
    material_type: 'Stainless Steel 316L',
    quality_grade: 'C',
    laser_power: 150.0,
    scan_speed: 1500.0,
    layer_thickness: 0.02,
    density: 0.92,
    surface_roughness: 8.5,
    status: 'completed',
    duration: 2800,
    energy_consumption: 120.5,
    powder_usage: 1.8,
    build_temperature: 75.0,
    chamber_pressure: 0.15,
    hatch_spacing: 0.15,
    exposure_time: 0.06
});

CREATE (p5:Process {
    process_id: 'PROC_005',
    timestamp: datetime('2024-01-17T11:00:00'),
    material_type: 'Ti6Al4V',
    quality_grade: 'A',
    laser_power: 220.0,
    scan_speed: 900.0,
    layer_thickness: 0.03,
    density: 0.97,
    surface_roughness: 5.0,
    status: 'in_progress',
    duration: 0,
    energy_consumption: 0.0,
    powder_usage: 0.0,
    build_temperature: 82.0,
    chamber_pressure: 0.1,
    hatch_spacing: 0.1,
    exposure_time: 0.1
});

// =============================================================================
// SAMPLE PARTS
// =============================================================================

// Create sample parts
CREATE (part1:Part {
    part_id: 'PART_001',
    part_type: 'turbine_blade',
    material_type: 'Ti6Al4V',
    dimensions: {x: 120.5, y: 45.2, z: 15.8},
    volume: 0.85,
    surface_area: 1250.3,
    weight: 3.2,
    status: 'completed',
    quality_grade: 'A',
    tolerance: 0.05,
    finish_quality: 'excellent'
});

CREATE (part2:Part {
    part_id: 'PART_002',
    part_type: 'engine_housing',
    material_type: 'Ti6Al4V',
    dimensions: {x: 200.0, y: 150.0, z: 80.0},
    volume: 2.4,
    surface_area: 3200.0,
    weight: 8.5,
    status: 'completed',
    quality_grade: 'B',
    tolerance: 0.08,
    finish_quality: 'good'
});

CREATE (part3:Part {
    part_id: 'PART_003',
    part_type: 'nozzle',
    material_type: 'Inconel 718',
    dimensions: {x: 50.0, y: 50.0, z: 100.0},
    volume: 0.25,
    surface_area: 800.0,
    weight: 2.0,
    status: 'completed',
    quality_grade: 'A',
    tolerance: 0.03,
    finish_quality: 'excellent'
});

CREATE (part4:Part {
    part_id: 'PART_004',
    part_type: 'bracket',
    material_type: 'Stainless Steel 316L',
    dimensions: {x: 80.0, y: 60.0, z: 20.0},
    volume: 0.096,
    surface_area: 400.0,
    weight: 0.8,
    status: 'completed',
    quality_grade: 'C',
    tolerance: 0.1,
    finish_quality: 'fair'
});

CREATE (part5:Part {
    part_id: 'PART_005',
    part_type: 'prototype_component',
    material_type: 'Ti6Al4V',
    dimensions: {x: 100.0, y: 100.0, z: 50.0},
    volume: 0.5,
    surface_area: 1000.0,
    weight: 2.2,
    status: 'in_progress',
    quality_grade: null,
    tolerance: 0.05,
    finish_quality: null
});

// =============================================================================
// SAMPLE QUALITY METRICS
// =============================================================================

// Create sample quality metrics
CREATE (q1:Quality {
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

CREATE (q2:Quality {
    grade: 'B',
    metrics: {
        density: 0.95,
        surface_roughness: 6.8,
        dimensional_accuracy: 35.0,
        tensile_strength: 880,
        yield_strength: 800
    },
    standards: ['ISO 2768'],
    inspector: 'John Doe',
    inspection_date: date('2024-01-16'),
    test_method: 'CT_scan',
    confidence_level: 0.90
});

CREATE (q3:Quality {
    grade: 'A',
    metrics: {
        density: 0.99,
        surface_roughness: 4.5,
        dimensional_accuracy: 20.0,
        tensile_strength: 1250,
        yield_strength: 1000
    },
    standards: ['ISO 2768', 'ASTM F2924'],
    inspector: 'John Doe',
    inspection_date: date('2024-01-17'),
    test_method: 'CT_scan',
    confidence_level: 0.98
});

CREATE (q4:Quality {
    grade: 'C',
    metrics: {
        density: 0.92,
        surface_roughness: 8.5,
        dimensional_accuracy: 50.0,
        tensile_strength: 550,
        yield_strength: 280
    },
    standards: ['ISO 2768'],
    inspector: 'John Doe',
    inspection_date: date('2024-01-17'),
    test_method: 'CT_scan',
    confidence_level: 0.85
});

// =============================================================================
// SAMPLE ALERTS
// =============================================================================

// Create sample alerts
CREATE (a1:Alert {
    alert_id: 'ALERT_001',
    severity: 'warning',
    status: 'resolved',
    timestamp: datetime('2024-01-15T11:30:00'),
    message: 'Temperature deviation detected',
    threshold: 85.0,
    actual_value: 87.5,
    resolution_time: 300,
    resolved_by: 'USER_001',
    resolution_notes: 'Adjusted build temperature'
});

CREATE (a2:Alert {
    alert_id: 'ALERT_002',
    severity: 'critical',
    status: 'active',
    timestamp: datetime('2024-01-16T10:15:00'),
    message: 'Laser power fluctuation',
    threshold: 200.0,
    actual_value: 180.0,
    resolution_time: null,
    resolved_by: null,
    resolution_notes: null
});

CREATE (a3:Alert {
    alert_id: 'ALERT_003',
    severity: 'info',
    status: 'resolved',
    timestamp: datetime('2024-01-17T09:45:00'),
    message: 'Powder level low',
    threshold: 10.0,
    actual_value: 8.5,
    resolution_time: 120,
    resolved_by: 'USER_001',
    resolution_notes: 'Refilled powder hopper'
});

// =============================================================================
// SAMPLE DEFECTS
// =============================================================================

// Create sample defects
CREATE (d1:Defect {
    defect_id: 'DEF_001',
    defect_type: 'porosity',
    severity: 'minor',
    location: {x: 10.5, y: 20.3, z: 5.2},
    size: 0.1,
    detection_method: 'CT_scan',
    confidence: 0.85,
    status: 'detected',
    timestamp: datetime('2024-01-15T15:00:00'),
    impact: 'cosmetic'
});

CREATE (d2:Defect {
    defect_id: 'DEF_002',
    defect_type: 'crack',
    severity: 'critical',
    location: {x: 50.0, y: 30.0, z: 10.0},
    size: 2.5,
    detection_method: 'visual_inspection',
    confidence: 0.95,
    status: 'detected',
    timestamp: datetime('2024-01-16T14:30:00'),
    impact: 'structural'
});

CREATE (d3:Defect {
    defect_id: 'DEF_003',
    defect_type: 'surface_roughness',
    severity: 'minor',
    location: {x: 0.0, y: 0.0, z: 0.0},
    size: 0.0,
    detection_method: 'surface_profilometer',
    confidence: 0.90,
    status: 'detected',
    timestamp: datetime('2024-01-17T11:20:00'),
    impact: 'cosmetic'
});

// =============================================================================
// SAMPLE IMAGES
// =============================================================================

// Create sample images
CREATE (i1:Image {
    image_id: 'IMG_001',
    image_type: 'process_monitoring',
    format: 'PNG',
    resolution: {width: 1920, height: 1080},
    file_size: 2048576,
    timestamp: datetime('2024-01-15T10:45:00'),
    camera_position: {x: 10.5, y: 20.3, z: 2.1},
    lighting_conditions: 'standard',
    quality_score: 0.92,
    file_path: '/images/process_001.png'
});

CREATE (i2:Image {
    image_id: 'IMG_002',
    image_type: 'quality_inspection',
    format: 'JPEG',
    resolution: {width: 2560, height: 1440},
    file_size: 3145728,
    timestamp: datetime('2024-01-16T15:00:00'),
    camera_position: {x: 5.0, y: 10.0, z: 1.5},
    lighting_conditions: 'high_contrast',
    quality_score: 0.95,
    file_path: '/images/inspection_001.jpg'
});

// =============================================================================
// SAMPLE LOGS
// =============================================================================

// Create sample logs
CREATE (l1:Log {
    log_id: 'LOG_001',
    level: 'INFO',
    source: 'process_monitor',
    message: 'Process started successfully',
    timestamp: datetime('2024-01-15T10:30:00'),
    component: 'laser_control',
    session_id: 'SESSION_001',
    user_id: 'USER_001'
});

CREATE (l2:Log {
    log_id: 'LOG_002',
    level: 'WARNING',
    source: 'temperature_sensor',
    message: 'Temperature exceeded threshold',
    timestamp: datetime('2024-01-15T11:30:00'),
    component: 'temperature_control',
    session_id: 'SESSION_001',
    user_id: 'USER_001'
});

CREATE (l3:Log {
    log_id: 'LOG_003',
    level: 'ERROR',
    source: 'laser_system',
    message: 'Laser power fluctuation detected',
    timestamp: datetime('2024-01-16T10:15:00'),
    component: 'laser_control',
    session_id: 'SESSION_002',
    user_id: 'USER_002'
});

// =============================================================================
// CREATE RELATIONSHIPS
// =============================================================================

// Process relationships
MATCH (p1:Process {process_id: 'PROC_001'}), (m1:Machine {machine_id: 'MACHINE_001'})
CREATE (p1)-[:USES_MACHINE {duration: 3600, start_time: datetime('2024-01-15T10:30:00')}]->(m1);

MATCH (p1:Process {process_id: 'PROC_001'}), (part1:Part {part_id: 'PART_001'})
CREATE (p1)-[:CREATES_PART {quantity: 1, success_rate: 0.95}]->(part1);

MATCH (p1:Process {process_id: 'PROC_001'}), (b1:Build {build_id: 'BUILD_001'})
CREATE (p1)-[:PART_OF_BUILD {sequence: 1, priority: 'high'}]->(b1);

MATCH (p1:Process {process_id: 'PROC_001'}), (mat1:Material {material_type: 'Ti6Al4V'})
CREATE (p1)-[:USES_MATERIAL {quantity: 2.3, unit: 'kg'}]->(mat1);

MATCH (p1:Process {process_id: 'PROC_001'}), (q1:Quality {grade: 'A'})
CREATE (p1)-[:HAS_QUALITY {measured_at: datetime('2024-01-15T14:30:00')}]->(q1);

MATCH (p1:Process {process_id: 'PROC_001'}), (s1:Sensor {sensor_id: 'TEMP_001'})
CREATE (p1)-[:MONITORED_BY {sampling_rate: 1.0, active: true}]->(s1);

MATCH (p1:Process {process_id: 'PROC_001'}), (o1:Operator {operator_id: 'OP_001'})
CREATE (p1)-[:OPERATED_BY {shift: 'day', experience_level: 'expert'}]->(o1);

MATCH (p1:Process {process_id: 'PROC_001'}), (a1:Alert {alert_id: 'ALERT_001'})
CREATE (p1)-[:GENERATES_ALERT {triggered_at: datetime('2024-01-15T11:30:00')}]->(a1);

MATCH (p1:Process {process_id: 'PROC_001'}), (d1:Defect {defect_id: 'DEF_001'})
CREATE (p1)-[:HAS_DEFECT {detected_at: datetime('2024-01-15T15:00:00')}]->(d1);

MATCH (p1:Process {process_id: 'PROC_001'}), (i1:Image {image_id: 'IMG_001'})
CREATE (p1)-[:CAPTURED_BY {timestamp: datetime('2024-01-15T10:45:00')}]->(i1);

MATCH (p1:Process {process_id: 'PROC_001'}), (l1:Log {log_id: 'LOG_001'})
CREATE (p1)-[:LOGGED_IN {level: 'INFO', timestamp: datetime('2024-01-15T10:30:00')}]->(l1);

// Machine relationships
MATCH (m1:Machine {machine_id: 'MACHINE_001'}), (s1:Sensor {sensor_id: 'TEMP_001'})
CREATE (m1)-[:HAS_SENSOR {installation_date: date('2023-01-15')}]->(s1);

MATCH (m1:Machine {machine_id: 'MACHINE_001'}), (o1:Operator {operator_id: 'OP_001'})
CREATE (m1)-[:OPERATED_BY {authorization_level: 'full'}]->(o1);

// Part relationships
MATCH (part1:Part {part_id: 'PART_001'}), (b1:Build {build_id: 'BUILD_001'})
CREATE (part1)-[:BELONGS_TO_BUILD {sequence: 1, priority: 'high'}]->(b1);

MATCH (part1:Part {part_id: 'PART_001'}), (mat1:Material {material_type: 'Ti6Al4V'})
CREATE (part1)-[:MADE_OF_MATERIAL {quantity: 2.3, unit: 'kg'}]->(mat1);

MATCH (part1:Part {part_id: 'PART_001'}), (q1:Quality {grade: 'A'})
CREATE (part1)-[:HAS_QUALITY {measured_at: datetime('2024-01-16T09:00:00')}]->(q1);

MATCH (part1:Part {part_id: 'PART_001'}), (d1:Defect {defect_id: 'DEF_001'})
CREATE (part1)-[:HAS_DEFECT {detected_at: datetime('2024-01-15T15:00:00')}]->(d1);

// Build relationships
MATCH (b1:Build {build_id: 'BUILD_001'}), (u1:User {user_id: 'USER_001'})
CREATE (b1)-[:CREATED_BY {created_at: datetime('2024-01-15T08:00:00')}]->(u1);

// Sensor relationships
MATCH (s1:Sensor {sensor_id: 'TEMP_001'}), (a1:Alert {alert_id: 'ALERT_001'})
CREATE (s1)-[:TRIGGERS_ALERT {threshold_exceeded: true}]->(a1);

// User relationships
MATCH (u1:User {user_id: 'USER_001'}), (a1:Alert {alert_id: 'ALERT_001'})
CREATE (u1)-[:RESOLVES_ALERT {response_time: 300}]->(a1);

// Quality relationships
MATCH (q1:Quality {grade: 'A'}), (s1:Sensor {sensor_id: 'TEMP_001'})
CREATE (q1)-[:MEASURED_BY {instrument: 'CT_scanner'}]->(s1);

// Defect relationships
MATCH (d1:Defect {defect_id: 'DEF_001'}), (s1:Sensor {sensor_id: 'TEMP_001'})
CREATE (d1)-[:DETECTED_BY_SENSOR {detection_confidence: 0.85}]->(s1);

// Image relationships
MATCH (i1:Image {image_id: 'IMG_001'}), (part1:Part {part_id: 'PART_001'})
CREATE (i1)-[:SHOWS_PART {clarity: 0.92, angle: 'top_view'}]->(part1);

// Log relationships
MATCH (l1:Log {log_id: 'LOG_001'}), (u1:User {user_id: 'USER_001'})
CREATE (l1)-[:RECORDED_BY {timestamp: datetime('2024-01-15T10:30:00')}]->(u1);
