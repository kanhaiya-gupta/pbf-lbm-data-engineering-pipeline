// =============================================================================
// Neo4j Constraints and Indexes for Performance Optimization
// Comprehensive indexing strategy for PBF-LB/M manufacturing knowledge graph
// =============================================================================

// =============================================================================
// UNIQUE CONSTRAINTS
// =============================================================================

// Core Manufacturing Constraints
CREATE CONSTRAINT process_id_unique IF NOT EXISTS FOR (p:Process) REQUIRE p.process_id IS UNIQUE;
CREATE CONSTRAINT machine_id_unique IF NOT EXISTS FOR (m:Machine) REQUIRE m.machine_id IS UNIQUE;
CREATE CONSTRAINT part_id_unique IF NOT EXISTS FOR (p:Part) REQUIRE p.part_id IS UNIQUE;
CREATE CONSTRAINT build_id_unique IF NOT EXISTS FOR (b:Build) REQUIRE b.build_id IS UNIQUE;

// Material and Quality Constraints
CREATE CONSTRAINT material_type_unique IF NOT EXISTS FOR (m:Material) REQUIRE m.material_type IS UNIQUE;
CREATE CONSTRAINT batch_id_unique IF NOT EXISTS FOR (b:Batch) REQUIRE b.batch_id IS UNIQUE;
CREATE CONSTRAINT quality_grade_unique IF NOT EXISTS FOR (q:Quality) REQUIRE q.grade IS UNIQUE;

// Sensor and Monitoring Constraints
CREATE CONSTRAINT sensor_id_unique IF NOT EXISTS FOR (s:Sensor) REQUIRE s.sensor_id IS UNIQUE;
CREATE CONSTRAINT alert_id_unique IF NOT EXISTS FOR (a:Alert) REQUIRE a.alert_id IS UNIQUE;
CREATE CONSTRAINT measurement_id_unique IF NOT EXISTS FOR (m:Measurement) REQUIRE m.measurement_id IS UNIQUE;

// User and Operator Constraints
CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE;
CREATE CONSTRAINT operator_id_unique IF NOT EXISTS FOR (o:Operator) REQUIRE o.operator_id IS UNIQUE;

// File and Documentation Constraints
CREATE CONSTRAINT image_id_unique IF NOT EXISTS FOR (i:Image) REQUIRE i.image_id IS UNIQUE;
CREATE CONSTRAINT log_id_unique IF NOT EXISTS FOR (l:Log) REQUIRE l.log_id IS UNIQUE;
CREATE CONSTRAINT build_file_id_unique IF NOT EXISTS FOR (bf:BuildFile) REQUIRE bf.build_file_id IS UNIQUE;

// Defect and Quality Control Constraints
CREATE CONSTRAINT defect_id_unique IF NOT EXISTS FOR (d:Defect) REQUIRE d.defect_id IS UNIQUE;
CREATE CONSTRAINT inspection_id_unique IF NOT EXISTS FOR (i:Inspection) REQUIRE i.inspection_id IS UNIQUE;

// =============================================================================
// EXISTENCE CONSTRAINTS
// =============================================================================

// Process Constraints
CREATE CONSTRAINT process_timestamp_exists IF NOT EXISTS FOR (p:Process) REQUIRE p.timestamp IS NOT NULL;
CREATE CONSTRAINT process_status_exists IF NOT EXISTS FOR (p:Process) REQUIRE p.status IS NOT NULL;

// Machine Constraints
CREATE CONSTRAINT machine_status_exists IF NOT EXISTS FOR (m:Machine) REQUIRE m.status IS NOT NULL;
CREATE CONSTRAINT machine_type_exists IF NOT EXISTS FOR (m:Machine) REQUIRE m.machine_type IS NOT NULL;

// Part Constraints
CREATE CONSTRAINT part_type_exists IF NOT EXISTS FOR (p:Part) REQUIRE p.part_type IS NOT NULL;
CREATE CONSTRAINT part_status_exists IF NOT EXISTS FOR (p:Part) REQUIRE p.status IS NOT NULL;

// Build Constraints
CREATE CONSTRAINT build_name_exists IF NOT EXISTS FOR (b:Build) REQUIRE b.build_name IS NOT NULL;
CREATE CONSTRAINT build_status_exists IF NOT EXISTS FOR (b:Build) REQUIRE b.status IS NOT NULL;

// Material Constraints
CREATE CONSTRAINT material_type_exists IF NOT EXISTS FOR (m:Material) REQUIRE m.material_type IS NOT NULL;
CREATE CONSTRAINT batch_condition_exists IF NOT EXISTS FOR (b:Batch) REQUIRE b.condition IS NOT NULL;

// Quality Constraints
CREATE CONSTRAINT quality_grade_exists IF NOT EXISTS FOR (q:Quality) REQUIRE q.grade IS NOT NULL;
CREATE CONSTRAINT quality_metrics_exists IF NOT EXISTS FOR (q:Quality) REQUIRE q.metrics IS NOT NULL;

// Sensor Constraints
CREATE CONSTRAINT sensor_type_exists IF NOT EXISTS FOR (s:Sensor) REQUIRE s.sensor_type IS NOT NULL;
CREATE CONSTRAINT sensor_location_exists IF NOT EXISTS FOR (s:Sensor) REQUIRE s.location IS NOT NULL;

// Alert Constraints
CREATE CONSTRAINT alert_severity_exists IF NOT EXISTS FOR (a:Alert) REQUIRE a.severity IS NOT NULL;
CREATE CONSTRAINT alert_status_exists IF NOT EXISTS FOR (a:Alert) REQUIRE a.status IS NOT NULL;

// User Constraints
CREATE CONSTRAINT user_username_exists IF NOT EXISTS FOR (u:User) REQUIRE u.username IS NOT NULL;
CREATE CONSTRAINT user_role_exists IF NOT EXISTS FOR (u:User) REQUIRE u.role IS NOT NULL;

// Operator Constraints
CREATE CONSTRAINT operator_name_exists IF NOT EXISTS FOR (o:Operator) REQUIRE o.name IS NOT NULL;
CREATE CONSTRAINT operator_certification_exists IF NOT EXISTS FOR (o:Operator) REQUIRE o.certification IS NOT NULL;

// =============================================================================
// PERFORMANCE INDEXES
// =============================================================================

// Process Performance Indexes
CREATE INDEX process_timestamp_index IF NOT EXISTS FOR (p:Process) ON (p.timestamp);
CREATE INDEX process_material_type_index IF NOT EXISTS FOR (p:Process) ON (p.material_type);
CREATE INDEX process_quality_grade_index IF NOT EXISTS FOR (p:Process) ON (p.quality_grade);
CREATE INDEX process_status_index IF NOT EXISTS FOR (p:Process) ON (p.status);
CREATE INDEX process_laser_power_index IF NOT EXISTS FOR (p:Process) ON (p.laser_power);
CREATE INDEX process_scan_speed_index IF NOT EXISTS FOR (p:Process) ON (p.scan_speed);
CREATE INDEX process_layer_thickness_index IF NOT EXISTS FOR (p:Process) ON (p.layer_thickness);
CREATE INDEX process_density_index IF NOT EXISTS FOR (p:Process) ON (p.density);
CREATE INDEX process_surface_roughness_index IF NOT EXISTS FOR (p:Process) ON (p.surface_roughness);
CREATE INDEX process_duration_index IF NOT EXISTS FOR (p:Process) ON (p.duration);
CREATE INDEX process_energy_consumption_index IF NOT EXISTS FOR (p:Process) ON (p.energy_consumption);
CREATE INDEX process_build_temperature_index IF NOT EXISTS FOR (p:Process) ON (p.build_temperature);

// Machine Performance Indexes
CREATE INDEX machine_type_index IF NOT EXISTS FOR (m:Machine) ON (m.machine_type);
CREATE INDEX machine_status_index IF NOT EXISTS FOR (m:Machine) ON (m.status);
CREATE INDEX machine_location_index IF NOT EXISTS FOR (m:Machine) ON (m.location);
CREATE INDEX machine_model_index IF NOT EXISTS FOR (m:Machine) ON (m.model);
CREATE INDEX machine_installation_date_index IF NOT EXISTS FOR (m:Machine) ON (m.installation_date);
CREATE INDEX machine_laser_power_max_index IF NOT EXISTS FOR (m:Machine) ON (m.laser_power_max);
CREATE INDEX machine_accuracy_index IF NOT EXISTS FOR (m:Machine) ON (m.accuracy);

// Part Performance Indexes
CREATE INDEX part_type_index IF NOT EXISTS FOR (p:Part) ON (p.part_type);
CREATE INDEX part_status_index IF NOT EXISTS FOR (p:Part) ON (p.status);
CREATE INDEX part_material_type_index IF NOT EXISTS FOR (p:Part) ON (p.material_type);
CREATE INDEX part_quality_grade_index IF NOT EXISTS FOR (p:Part) ON (p.quality_grade);
CREATE INDEX part_volume_index IF NOT EXISTS FOR (p:Part) ON (p.volume);
CREATE INDEX part_surface_area_index IF NOT EXISTS FOR (p:Part) ON (p.surface_area);
CREATE INDEX part_weight_index IF NOT EXISTS FOR (p:Part) ON (p.weight);
CREATE INDEX part_tolerance_index IF NOT EXISTS FOR (p:Part) ON (p.tolerance);

// Build Performance Indexes
CREATE INDEX build_name_index IF NOT EXISTS FOR (b:Build) ON (b.build_name);
CREATE INDEX build_status_index IF NOT EXISTS FOR (b:Build) ON (b.status);
CREATE INDEX build_created_date_index IF NOT EXISTS FOR (b:Build) ON (b.created_date);
CREATE INDEX build_completed_date_index IF NOT EXISTS FOR (b:Build) ON (b.completed_date);
CREATE INDEX build_total_parts_index IF NOT EXISTS FOR (b:Build) ON (b.total_parts);
CREATE INDEX build_success_rate_index IF NOT EXISTS FOR (b:Build) ON (b.success_rate);
CREATE INDEX build_total_duration_index IF NOT EXISTS FOR (b:Build) ON (b.total_duration);
CREATE INDEX build_material_usage_index IF NOT EXISTS FOR (b:Build) ON (b.material_usage);

// Material Performance Indexes
CREATE INDEX material_type_index IF NOT EXISTS FOR (m:Material) ON (m.material_type);
CREATE INDEX material_supplier_index IF NOT EXISTS FOR (m:Material) ON (m.supplier);
CREATE INDEX material_certification_index IF NOT EXISTS FOR (m:Material) ON (m.certification);
CREATE INDEX material_batch_number_index IF NOT EXISTS FOR (m:Material) ON (m.batch_number);
CREATE INDEX material_condition_index IF NOT EXISTS FOR (m:Material) ON (m.condition);
CREATE INDEX material_storage_temperature_index IF NOT EXISTS FOR (m:Material) ON (m.storage_temperature);
CREATE INDEX material_humidity_index IF NOT EXISTS FOR (m:Material) ON (m.humidity);

// Quality Performance Indexes
CREATE INDEX quality_grade_index IF NOT EXISTS FOR (q:Quality) ON (q.grade);
CREATE INDEX quality_inspector_index IF NOT EXISTS FOR (q:Quality) ON (q.inspector);
CREATE INDEX quality_inspection_date_index IF NOT EXISTS FOR (q:Quality) ON (q.inspection_date);
CREATE INDEX quality_test_method_index IF NOT EXISTS FOR (q:Quality) ON (q.test_method);
CREATE INDEX quality_confidence_level_index IF NOT EXISTS FOR (q:Quality) ON (q.confidence_level);

// Sensor Performance Indexes
CREATE INDEX sensor_type_index IF NOT EXISTS FOR (s:Sensor) ON (s.sensor_type);
CREATE INDEX sensor_location_index IF NOT EXISTS FOR (s:Sensor) ON (s.location);
CREATE INDEX sensor_model_index IF NOT EXISTS FOR (s:Sensor) ON (s.model);
CREATE INDEX sensor_calibration_date_index IF NOT EXISTS FOR (s:Sensor) ON (s.calibration_date);
CREATE INDEX sensor_accuracy_index IF NOT EXISTS FOR (s:Sensor) ON (s.accuracy);
CREATE INDEX sensor_sampling_rate_index IF NOT EXISTS FOR (s:Sensor) ON (s.sampling_rate);
CREATE INDEX sensor_status_index IF NOT EXISTS FOR (s:Sensor) ON (s.status);

// Alert Performance Indexes
CREATE INDEX alert_severity_index IF NOT EXISTS FOR (a:Alert) ON (a.severity);
CREATE INDEX alert_status_index IF NOT EXISTS FOR (a:Alert) ON (a.status);
CREATE INDEX alert_timestamp_index IF NOT EXISTS FOR (a:Alert) ON (a.timestamp);
CREATE INDEX alert_threshold_index IF NOT EXISTS FOR (a:Alert) ON (a.threshold);
CREATE INDEX alert_actual_value_index IF NOT EXISTS FOR (a:Alert) ON (a.actual_value);
CREATE INDEX alert_resolution_time_index IF NOT EXISTS FOR (a:Alert) ON (a.resolution_time);

// User Performance Indexes
CREATE INDEX user_username_index IF NOT EXISTS FOR (u:User) ON (u.username);
CREATE INDEX user_role_index IF NOT EXISTS FOR (u:User) ON (u.role);
CREATE INDEX user_department_index IF NOT EXISTS FOR (u:User) ON (u.department);
CREATE INDEX user_email_index IF NOT EXISTS FOR (u:User) ON (u.email);
CREATE INDEX user_active_index IF NOT EXISTS FOR (u:User) ON (u.active);
CREATE INDEX user_last_login_index IF NOT EXISTS FOR (u:User) ON (u.last_login);

// Operator Performance Indexes
CREATE INDEX operator_name_index IF NOT EXISTS FOR (o:Operator) ON (o.name);
CREATE INDEX operator_certification_index IF NOT EXISTS FOR (o:Operator) ON (o.certification);
CREATE INDEX operator_experience_years_index IF NOT EXISTS FOR (o:Operator) ON (o.experience_years);
CREATE INDEX operator_shift_index IF NOT EXISTS FOR (o:Operator) ON (o.shift);

// Image Performance Indexes
CREATE INDEX image_type_index IF NOT EXISTS FOR (i:Image) ON (i.image_type);
CREATE INDEX image_format_index IF NOT EXISTS FOR (i:Image) ON (i.format);
CREATE INDEX image_timestamp_index IF NOT EXISTS FOR (i:Image) ON (i.timestamp);
CREATE INDEX image_file_size_index IF NOT EXISTS FOR (i:Image) ON (i.file_size);
CREATE INDEX image_quality_score_index IF NOT EXISTS FOR (i:Image) ON (i.quality_score);

// Log Performance Indexes
CREATE INDEX log_level_index IF NOT EXISTS FOR (l:Log) ON (l.level);
CREATE INDEX log_source_index IF NOT EXISTS FOR (l:Log) ON (l.source);
CREATE INDEX log_timestamp_index IF NOT EXISTS FOR (l:Log) ON (l.timestamp);
CREATE INDEX log_component_index IF NOT EXISTS FOR (l:Log) ON (l.component);
CREATE INDEX log_session_id_index IF NOT EXISTS FOR (l:Log) ON (l.session_id);
CREATE INDEX log_user_id_index IF NOT EXISTS FOR (l:Log) ON (l.user_id);

// Defect Performance Indexes
CREATE INDEX defect_type_index IF NOT EXISTS FOR (d:Defect) ON (d.defect_type);
CREATE INDEX defect_severity_index IF NOT EXISTS FOR (d:Defect) ON (d.severity);
CREATE INDEX defect_detection_method_index IF NOT EXISTS FOR (d:Defect) ON (d.detection_method);
CREATE INDEX defect_confidence_index IF NOT EXISTS FOR (d:Defect) ON (d.confidence);
CREATE INDEX defect_status_index IF NOT EXISTS FOR (d:Defect) ON (d.status);
CREATE INDEX defect_timestamp_index IF NOT EXISTS FOR (d:Defect) ON (d.timestamp);

// =============================================================================
// COMPOSITE INDEXES FOR COMPLEX QUERIES
// =============================================================================

// Process Composite Indexes
CREATE INDEX process_material_status_index IF NOT EXISTS FOR (p:Process) ON (p.material_type, p.status);
CREATE INDEX process_quality_status_index IF NOT EXISTS FOR (p:Process) ON (p.quality_grade, p.status);
CREATE INDEX process_timestamp_status_index IF NOT EXISTS FOR (p:Process) ON (p.timestamp, p.status);
CREATE INDEX process_laser_scan_index IF NOT EXISTS FOR (p:Process) ON (p.laser_power, p.scan_speed);

// Machine Composite Indexes
CREATE INDEX machine_type_status_index IF NOT EXISTS FOR (m:Machine) ON (m.machine_type, m.status);
CREATE INDEX machine_location_status_index IF NOT EXISTS FOR (m:Machine) ON (m.location, m.status);

// Part Composite Indexes
CREATE INDEX part_type_status_index IF NOT EXISTS FOR (p:Part) ON (p.part_type, p.status);
CREATE INDEX part_material_quality_index IF NOT EXISTS FOR (p:Part) ON (p.material_type, p.quality_grade);

// Build Composite Indexes
CREATE INDEX build_status_date_index IF NOT EXISTS FOR (b:Build) ON (b.status, b.created_date);
CREATE INDEX build_success_parts_index IF NOT EXISTS FOR (b:Build) ON (b.success_rate, b.total_parts);

// Alert Composite Indexes
CREATE INDEX alert_severity_status_index IF NOT EXISTS FOR (a:Alert) ON (a.severity, a.status);
CREATE INDEX alert_timestamp_status_index IF NOT EXISTS FOR (a:Alert) ON (a.timestamp, a.status);

// User Composite Indexes
CREATE INDEX user_role_department_index IF NOT EXISTS FOR (u:User) ON (u.role, u.department);
CREATE INDEX user_active_login_index IF NOT EXISTS FOR (u:User) ON (u.active, u.last_login);

// =============================================================================
// FULL-TEXT INDEXES FOR SEARCH
// =============================================================================

// Process Full-Text Indexes
CREATE FULLTEXT INDEX process_fulltext IF NOT EXISTS FOR (p:Process) ON EACH [p.process_id, p.material_type, p.status];

// Machine Full-Text Indexes
CREATE FULLTEXT INDEX machine_fulltext IF NOT EXISTS FOR (m:Machine) ON EACH [m.machine_id, m.machine_type, m.model, m.location];

// Part Full-Text Indexes
CREATE FULLTEXT INDEX part_fulltext IF NOT EXISTS FOR (p:Part) ON EACH [p.part_id, p.part_type, p.material_type];

// Build Full-Text Indexes
CREATE FULLTEXT INDEX build_fulltext IF NOT EXISTS FOR (b:Build) ON EACH [b.build_id, b.build_name, b.status];

// Material Full-Text Indexes
CREATE FULLTEXT INDEX material_fulltext IF NOT EXISTS FOR (m:Material) ON EACH [m.material_type, m.supplier, m.certification];

// User Full-Text Indexes
CREATE FULLTEXT INDEX user_fulltext IF NOT EXISTS FOR (u:User) ON EACH [u.username, u.name, u.email, u.department];

// Log Full-Text Indexes
CREATE FULLTEXT INDEX log_fulltext IF NOT EXISTS FOR (l:Log) ON EACH [l.message, l.source, l.component];

// =============================================================================
// RELATIONSHIP INDEXES
// =============================================================================

// Process Relationship Indexes
CREATE INDEX process_uses_machine_index IF NOT EXISTS FOR ()-[r:USES_MACHINE]-() ON (r.duration, r.start_time);
CREATE INDEX process_creates_part_index IF NOT EXISTS FOR ()-[r:CREATES_PART]-() ON (r.quantity, r.success_rate);
CREATE INDEX process_uses_material_index IF NOT EXISTS FOR ()-[r:USES_MATERIAL]-() ON (r.quantity, r.unit);
CREATE INDEX process_has_quality_index IF NOT EXISTS FOR ()-[r:HAS_QUALITY]-() ON (r.measured_at);
CREATE INDEX process_monitored_by_index IF NOT EXISTS FOR ()-[r:MONITORED_BY]-() ON (r.sampling_rate, r.active);

// Machine Relationship Indexes
CREATE INDEX machine_hosts_process_index IF NOT EXISTS FOR ()-[r:HOSTS_PROCESS]-() ON (r.capacity, r.utilization);
CREATE INDEX machine_has_sensor_index IF NOT EXISTS FOR ()-[r:HAS_SENSOR]-() ON (r.installation_date);

// Part Relationship Indexes
CREATE INDEX part_belongs_to_build_index IF NOT EXISTS FOR ()-[r:BELONGS_TO_BUILD]-() ON (r.sequence, r.priority);
CREATE INDEX part_created_by_process_index IF NOT EXISTS FOR ()-[r:CREATED_BY_PROCESS]-() ON (r.creation_time);

// Quality Relationship Indexes
CREATE INDEX quality_measured_by_index IF NOT EXISTS FOR ()-[r:MEASURED_BY]-() ON (r.instrument);
CREATE INDEX quality_influences_process_index IF NOT EXISTS FOR ()-[r:INFLUENCES_PROCESS]-() ON (r.correlation);

// Sensor Relationship Indexes
CREATE INDEX sensor_monitors_process_index IF NOT EXISTS FOR ()-[r:MONITORS_PROCESS]-() ON (r.coverage, r.accuracy);
CREATE INDEX sensor_triggers_alert_index IF NOT EXISTS FOR ()-[r:TRIGGERS_ALERT]-() ON (r.threshold_exceeded);

// Alert Relationship Indexes
CREATE INDEX alert_affects_process_index IF NOT EXISTS FOR ()-[r:AFFECTS_PROCESS]-() ON (r.severity, r.resolved);
CREATE INDEX alert_notifies_user_index IF NOT EXISTS FOR ()-[r:NOTIFIES_USER]-() ON (r.notification_time);

// User Relationship Indexes
CREATE INDEX user_operates_machine_index IF NOT EXISTS FOR ()-[r:OPERATES_MACHINE]-() ON (r.authorization_level);
CREATE INDEX user_manages_process_index IF NOT EXISTS FOR ()-[r:MANAGES_PROCESS]-() ON (r.responsibility);

// =============================================================================
// PERFORMANCE MONITORING QUERIES
// =============================================================================

// Query to check index usage
CALL db.indexes() YIELD name, state, type, labelsOrTypes, properties
RETURN name, state, type, labelsOrTypes, properties
ORDER BY name;

// Query to check constraint usage
CALL db.constraints() YIELD name, type, labelsOrTypes, properties
RETURN name, type, labelsOrTypes, properties
ORDER BY name;

// Query to check database statistics
CALL db.stats() YIELD nodes, relationships, properties, labels, relTypes
RETURN nodes, relationships, properties, labels, relTypes;
