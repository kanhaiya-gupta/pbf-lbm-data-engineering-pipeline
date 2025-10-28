// =============================================================================
// Pre-built Analytical Queries for PBF-LB/M Manufacturing Knowledge Graph
// Comprehensive analytics queries for process optimization and quality analysis
// =============================================================================

// =============================================================================
// PROCESS ANALYTICS QUERIES
// =============================================================================

// Query 1: Process Performance Analysis
// Find processes with highest quality grades and their parameters
MATCH (p:Process)-[:HAS_QUALITY]->(q:Quality)
WHERE q.grade = 'A'
RETURN p.process_id, p.material_type, p.laser_power, p.scan_speed, 
       p.layer_thickness, p.density, p.surface_roughness, q.metrics
ORDER BY p.density DESC, p.surface_roughness ASC;

// Query 2: Process Efficiency Analysis
// Calculate process efficiency metrics
MATCH (p:Process)
RETURN p.material_type,
       count(p) as total_processes,
       avg(p.duration) as avg_duration,
       avg(p.energy_consumption) as avg_energy,
       avg(p.powder_usage) as avg_powder_usage,
       avg(p.density) as avg_density,
       avg(p.surface_roughness) as avg_roughness
ORDER BY avg_density DESC;

// Query 3: Process Parameter Correlation
// Find correlation between laser power and quality
MATCH (p:Process)-[:HAS_QUALITY]->(q:Quality)
RETURN p.laser_power,
       avg(q.metrics.density) as avg_density,
       avg(q.metrics.surface_roughness) as avg_roughness,
       count(p) as process_count
ORDER BY p.laser_power;

// Query 4: Process Success Rate by Material
// Calculate success rates for different materials
MATCH (p:Process)-[:HAS_QUALITY]->(q:Quality)
RETURN p.material_type,
       count(p) as total_processes,
       sum(CASE WHEN q.grade = 'A' THEN 1 ELSE 0 END) as successful_processes,
       sum(CASE WHEN q.grade = 'A' THEN 1 ELSE 0 END) * 100.0 / count(p) as success_rate
ORDER BY success_rate DESC;

// =============================================================================
// MACHINE ANALYTICS QUERIES
// =============================================================================

// Query 5: Machine Utilization Analysis
// Find machines with highest utilization and success rates
MATCH (machine:Machine)<-[:USES_MACHINE]-(p:Process)-[:HAS_QUALITY]->(q:Quality)
RETURN machine.machine_id, machine.model, machine.location,
       count(p) as total_processes,
       sum(CASE WHEN q.grade = 'A' THEN 1 ELSE 0 END) as successful_processes,
       sum(CASE WHEN q.grade = 'A' THEN 1 ELSE 0 END) * 100.0 / count(p) as success_rate,
       avg(p.duration) as avg_process_duration,
       sum(p.energy_consumption) as total_energy_consumption
ORDER BY success_rate DESC, total_processes DESC;

// Query 6: Machine Performance by Type
// Compare performance across machine types
MATCH (machine:Machine)<-[:USES_MACHINE]-(p:Process)
RETURN machine.machine_type,
       count(p) as total_processes,
       avg(p.duration) as avg_duration,
       avg(p.energy_consumption) as avg_energy,
       avg(p.density) as avg_density,
       avg(p.surface_roughness) as avg_roughness
ORDER BY avg_density DESC;

// Query 7: Machine Maintenance Analysis
// Find machines with most alerts and issues
MATCH (machine:Machine)<-[:USES_MACHINE]-(p:Process)-[:GENERATES_ALERT]->(a:Alert)
RETURN machine.machine_id, machine.model,
       count(a) as total_alerts,
       sum(CASE WHEN a.severity = 'critical' THEN 1 ELSE 0 END) as critical_alerts,
       sum(CASE WHEN a.severity = 'warning' THEN 1 ELSE 0 END) as warning_alerts,
       avg(a.resolution_time) as avg_resolution_time
ORDER BY total_alerts DESC;

// =============================================================================
// QUALITY ANALYTICS QUERIES
// =============================================================================

// Query 8: Quality Trend Analysis
// Analyze quality trends over time
MATCH (p:Process)-[:HAS_QUALITY]->(q:Quality)
WHERE p.timestamp >= datetime('2024-01-01')
RETURN date(p.timestamp) as process_date,
       count(p) as total_processes,
       avg(q.metrics.density) as avg_density,
       avg(q.metrics.surface_roughness) as avg_roughness,
       sum(CASE WHEN q.grade = 'A' THEN 1 ELSE 0 END) as grade_a_count
ORDER BY process_date;

// Query 9: Quality Correlation Analysis
// Find correlations between process parameters and quality
MATCH (p:Process)-[:HAS_QUALITY]->(q:Quality)
RETURN p.laser_power, p.scan_speed, p.layer_thickness,
       avg(q.metrics.density) as avg_density,
       avg(q.metrics.surface_roughness) as avg_roughness,
       avg(q.metrics.dimensional_accuracy) as avg_accuracy,
       count(p) as sample_size
ORDER BY avg_density DESC;

// Query 10: Quality Standards Compliance
// Check compliance with quality standards
MATCH (p:Process)-[:HAS_QUALITY]->(q:Quality)
WHERE q.standards CONTAINS 'ISO 2768'
RETURN p.material_type,
       count(p) as total_processes,
       sum(CASE WHEN q.metrics.dimensional_accuracy <= 25.0 THEN 1 ELSE 0 END) as compliant_processes,
       sum(CASE WHEN q.metrics.dimensional_accuracy <= 25.0 THEN 1 ELSE 0 END) * 100.0 / count(p) as compliance_rate
ORDER BY compliance_rate DESC;

// =============================================================================
// SENSOR ANALYTICS QUERIES
// =============================================================================

// Query 11: Sensor Performance Analysis
// Analyze sensor performance and reliability
MATCH (s:Sensor)-[:MONITORS_PROCESS]->(p:Process)-[:GENERATES_ALERT]->(a:Alert)
RETURN s.sensor_id, s.sensor_type, s.location,
       count(p) as monitored_processes,
       count(a) as triggered_alerts,
       avg(s.accuracy) as sensor_accuracy,
       avg(s.sampling_rate) as avg_sampling_rate
ORDER BY monitored_processes DESC;

// Query 12: Sensor Coverage Analysis
// Analyze sensor coverage across processes
MATCH (s:Sensor)-[:MONITORS_PROCESS]->(p:Process)
RETURN s.sensor_type, s.location,
       count(p) as monitored_processes,
       avg(p.duration) as avg_process_duration,
       sum(p.energy_consumption) as total_energy_monitored
ORDER BY monitored_processes DESC;

// Query 13: Sensor Alert Correlation
// Find correlation between sensor readings and alerts
MATCH (s:Sensor)-[:TRIGGERS_ALERT]->(a:Alert)
WHERE a.timestamp >= datetime('2024-01-01')
RETURN s.sensor_type, s.location,
       count(a) as total_alerts,
       sum(CASE WHEN a.severity = 'critical' THEN 1 ELSE 0 END) as critical_alerts,
       avg(a.threshold) as avg_threshold,
       avg(a.actual_value) as avg_actual_value
ORDER BY total_alerts DESC;

// =============================================================================
// DEFECT ANALYTICS QUERIES
// =============================================================================

// Query 14: Defect Pattern Analysis
// Analyze defect patterns by material and process parameters
MATCH (p:Process)-[:HAS_DEFECT]->(d:Defect)
RETURN p.material_type, d.defect_type,
       count(d) as defect_count,
       avg(d.size) as avg_defect_size,
       sum(CASE WHEN d.severity = 'critical' THEN 1 ELSE 0 END) as critical_defects,
       sum(CASE WHEN d.severity = 'minor' THEN 1 ELSE 0 END) as minor_defects
ORDER BY defect_count DESC;

// Query 15: Defect Detection Analysis
// Analyze defect detection methods and effectiveness
MATCH (d:Defect)-[:DETECTED_BY_SENSOR]->(s:Sensor)
RETURN d.defect_type, s.sensor_type,
       count(d) as detected_defects,
       avg(d.confidence) as avg_detection_confidence,
       sum(CASE WHEN d.confidence >= 0.8 THEN 1 ELSE 0 END) as high_confidence_detections
ORDER BY detected_defects DESC;

// Query 16: Defect Impact Analysis
// Analyze impact of defects on parts and processes
MATCH (d:Defect)-[:AFFECTS_PART]->(part:Part)-[:CREATED_BY_PROCESS]->(p:Process)
RETURN d.defect_type, part.part_type,
       count(d) as defect_count,
       avg(d.size) as avg_defect_size,
       avg(p.density) as avg_part_density,
       avg(p.surface_roughness) as avg_part_roughness
ORDER BY defect_count DESC;

// =============================================================================
// BUILD ANALYTICS QUERIES
// =============================================================================

// Query 17: Build Success Analysis
// Analyze build success rates and performance
MATCH (b:Build)-[:CONTAINS_PROCESS]->(p:Process)-[:HAS_QUALITY]->(q:Quality)
RETURN b.build_id, b.build_name,
       count(p) as total_processes,
       sum(CASE WHEN q.grade = 'A' THEN 1 ELSE 0 END) as successful_processes,
       sum(CASE WHEN q.grade = 'A' THEN 1 ELSE 0 END) * 100.0 / count(p) as success_rate,
       avg(b.total_duration) as avg_build_duration,
       avg(b.material_usage) as avg_material_usage
ORDER BY success_rate DESC;

// Query 18: Build Efficiency Analysis
// Analyze build efficiency metrics
MATCH (b:Build)
RETURN b.build_name,
       count(b) as total_builds,
       avg(b.total_parts) as avg_parts_per_build,
       avg(b.success_rate) as avg_success_rate,
       avg(b.total_duration) as avg_duration,
       avg(b.material_usage) as avg_material_usage
ORDER BY avg_success_rate DESC;

// =============================================================================
// OPERATOR ANALYTICS QUERIES
// =============================================================================

// Query 19: Operator Performance Analysis
// Analyze operator performance and experience
MATCH (o:Operator)-[:OPERATES_PROCESS]->(p:Process)-[:HAS_QUALITY]->(q:Quality)
RETURN o.operator_id, o.name, o.certification, o.experience_years,
       count(p) as total_processes,
       sum(CASE WHEN q.grade = 'A' THEN 1 ELSE 0 END) as successful_processes,
       sum(CASE WHEN q.grade = 'A' THEN 1 ELSE 0 END) * 100.0 / count(p) as success_rate,
       avg(p.duration) as avg_process_duration
ORDER BY success_rate DESC;

// Query 20: Operator Experience Correlation
// Find correlation between operator experience and process quality
MATCH (o:Operator)-[:OPERATES_PROCESS]->(p:Process)-[:HAS_QUALITY]->(q:Quality)
RETURN o.experience_years,
       count(p) as total_processes,
       avg(q.metrics.density) as avg_density,
       avg(q.metrics.surface_roughness) as avg_roughness,
       sum(CASE WHEN q.grade = 'A' THEN 1 ELSE 0 END) * 100.0 / count(p) as success_rate
ORDER BY o.experience_years;

// =============================================================================
// MATERIAL ANALYTICS QUERIES
// =============================================================================

// Query 21: Material Performance Analysis
// Analyze material performance across processes
MATCH (m:Material)<-[:USES_MATERIAL]-(p:Process)-[:HAS_QUALITY]->(q:Quality)
RETURN m.material_type, m.supplier,
       count(p) as total_processes,
       avg(q.metrics.density) as avg_density,
       avg(q.metrics.surface_roughness) as avg_roughness,
       sum(CASE WHEN q.grade = 'A' THEN 1 ELSE 0 END) * 100.0 / count(p) as success_rate
ORDER BY success_rate DESC;

// Query 22: Material Batch Analysis
// Analyze material batch performance
MATCH (m:Material)-[:CONTAINS_BATCH]->(b:Batch)-[:USED_IN_PROCESS]->(p:Process)-[:HAS_QUALITY]->(q:Quality)
RETURN m.material_type, b.batch_id, b.condition,
       count(p) as total_processes,
       avg(q.metrics.density) as avg_density,
       sum(CASE WHEN q.grade = 'A' THEN 1 ELSE 0 END) * 100.0 / count(p) as success_rate
ORDER BY success_rate DESC;

// =============================================================================
// ALERT ANALYTICS QUERIES
// =============================================================================

// Query 23: Alert Pattern Analysis
// Analyze alert patterns and resolution times
MATCH (a:Alert)
WHERE a.timestamp >= datetime('2024-01-01')
RETURN a.severity, a.status,
       count(a) as total_alerts,
       avg(a.resolution_time) as avg_resolution_time,
       sum(CASE WHEN a.resolution_time IS NOT NULL THEN 1 ELSE 0 END) as resolved_alerts
ORDER BY total_alerts DESC;

// Query 24: Alert Response Analysis
// Analyze alert response times and resolution
MATCH (a:Alert)-[:AFFECTS_PROCESS]->(p:Process)
RETURN a.severity,
       count(a) as total_alerts,
       avg(a.resolution_time) as avg_resolution_time,
       sum(CASE WHEN a.resolution_time IS NOT NULL THEN 1 ELSE 0 END) as resolved_alerts,
       sum(CASE WHEN a.resolution_time IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / count(a) as resolution_rate
ORDER BY total_alerts DESC;

// =============================================================================
// COMPREHENSIVE ANALYTICS QUERIES
// =============================================================================

// Query 25: Manufacturing KPI Dashboard
// Comprehensive manufacturing KPIs
MATCH (p:Process)-[:HAS_QUALITY]->(q:Quality)
WHERE p.timestamp >= datetime('2024-01-01')
RETURN 
    count(p) as total_processes,
    sum(CASE WHEN q.grade = 'A' THEN 1 ELSE 0 END) as successful_processes,
    sum(CASE WHEN q.grade = 'A' THEN 1 ELSE 0 END) * 100.0 / count(p) as overall_success_rate,
    avg(p.duration) as avg_process_duration,
    avg(p.energy_consumption) as avg_energy_consumption,
    avg(q.metrics.density) as avg_density,
    avg(q.metrics.surface_roughness) as avg_roughness;

// Query 26: Process Optimization Recommendations
// Identify optimization opportunities
MATCH (p:Process)-[:HAS_QUALITY]->(q:Quality)
WHERE q.grade != 'A'
RETURN p.material_type,
       count(p) as failed_processes,
       avg(p.laser_power) as avg_laser_power,
       avg(p.scan_speed) as avg_scan_speed,
       avg(p.layer_thickness) as avg_layer_thickness,
       avg(q.metrics.density) as avg_density,
       avg(q.metrics.surface_roughness) as avg_roughness
ORDER BY failed_processes DESC;

// Query 27: Predictive Maintenance Indicators
// Identify machines and processes that may need maintenance
MATCH (machine:Machine)<-[:USES_MACHINE]-(p:Process)-[:GENERATES_ALERT]->(a:Alert)
WHERE a.timestamp >= datetime('2024-01-01')
RETURN machine.machine_id, machine.model,
       count(a) as total_alerts,
       sum(CASE WHEN a.severity = 'critical' THEN 1 ELSE 0 END) as critical_alerts,
       avg(a.resolution_time) as avg_resolution_time,
       sum(CASE WHEN a.resolution_time IS NULL THEN 1 ELSE 0 END) as unresolved_alerts
ORDER BY critical_alerts DESC, total_alerts DESC;
