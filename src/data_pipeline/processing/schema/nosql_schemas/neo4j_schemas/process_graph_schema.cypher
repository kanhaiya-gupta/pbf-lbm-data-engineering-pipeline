// Neo4j graph schema for PBF-LB/M manufacturing process relationships
// This schema defines nodes, relationships, and constraints for the manufacturing graph

// =============================================================================
// NODE LABELS AND CONSTRAINTS
// =============================================================================

// Process nodes
CREATE CONSTRAINT process_id_unique IF NOT EXISTS FOR (p:Process) REQUIRE p.process_id IS UNIQUE;
CREATE CONSTRAINT process_timestamp_exists IF NOT EXISTS FOR (p:Process) REQUIRE p.timestamp IS NOT NULL;

// Material nodes
CREATE CONSTRAINT material_type_unique IF NOT EXISTS FOR (m:Material) REQUIRE m.material_type IS UNIQUE;
CREATE CONSTRAINT batch_id_unique IF NOT EXISTS FOR (b:Batch) REQUIRE b.batch_id IS UNIQUE;

// Machine nodes
CREATE CONSTRAINT machine_id_unique IF NOT EXISTS FOR (m:Machine) REQUIRE m.machine_id IS UNIQUE;

// Quality nodes
CREATE CONSTRAINT quality_grade_exists IF NOT EXISTS FOR (q:Quality) REQUIRE q.grade IS NOT NULL;

// Sensor nodes
CREATE CONSTRAINT sensor_id_unique IF NOT EXISTS FOR (s:Sensor) REQUIRE s.sensor_id IS UNIQUE;

// Operator nodes
CREATE CONSTRAINT operator_id_unique IF NOT EXISTS FOR (o:Operator) REQUIRE o.operator_id IS UNIQUE;

// Defect nodes
CREATE CONSTRAINT defect_id_unique IF NOT EXISTS FOR (d:Defect) REQUIRE d.defect_id IS UNIQUE;

// =============================================================================
// NODE PROPERTIES AND INDEXES
// =============================================================================

// Process node indexes
CREATE INDEX process_timestamp_index IF NOT EXISTS FOR (p:Process) ON (p.timestamp);
CREATE INDEX process_material_type_index IF NOT EXISTS FOR (p:Process) ON (p.material_type);
CREATE INDEX process_quality_grade_index IF NOT EXISTS FOR (p:Process) ON (p.quality_grade);

// Material node indexes
CREATE INDEX material_properties_index IF NOT EXISTS FOR (m:Material) ON (m.properties);
CREATE INDEX batch_condition_index IF NOT EXISTS FOR (b:Batch) ON (b.condition);

// Machine node indexes
CREATE INDEX machine_type_index IF NOT EXISTS FOR (m:Machine) ON (m.machine_type);
CREATE INDEX machine_status_index IF NOT EXISTS FOR (m:Machine) ON (m.status);

// Quality node indexes
CREATE INDEX quality_metrics_index IF NOT EXISTS FOR (q:Quality) ON (q.metrics);
CREATE INDEX quality_grade_index IF NOT EXISTS FOR (q:Quality) ON (q.grade);

// Sensor node indexes
CREATE INDEX sensor_type_index IF NOT EXISTS FOR (s:Sensor) ON (s.sensor_type);
CREATE INDEX sensor_location_index IF NOT EXISTS FOR (s:Sensor) ON (s.location);

// =============================================================================
// RELATIONSHIP TYPES AND PROPERTIES
// =============================================================================

// Process relationships
// (Process)-[:USES_MATERIAL]->(Material)
// (Process)-[:EXECUTED_ON]->(Machine)
// (Process)-[:HAS_QUALITY]->(Quality)
// (Process)-[:MONITORED_BY]->(Sensor)
// (Process)-[:OPERATED_BY]->(Operator)
// (Process)-[:HAS_DEFECT]->(Defect)

// Material relationships
// (Material)-[:CONTAINS]->(Batch)
// (Batch)-[:USED_IN]->(Process)

// Machine relationships
// (Machine)-[:HAS_SENSOR]->(Sensor)
// (Machine)-[:EXECUTES]->(Process)

// Quality relationships
// (Quality)-[:MEASURED_BY]->(Sensor)
// (Quality)-[:INFLUENCES]->(Process)

// Defect relationships
// (Defect)-[:DETECTED_BY]->(Sensor)
// (Defect)-[:AFFECTS]->(Process)

// =============================================================================
// SAMPLE DATA CREATION QUERIES
// =============================================================================

// Create sample process node
CREATE (p:Process {
    process_id: 'PROC_001',
    timestamp: datetime('2024-01-15T10:30:00'),
    material_type: 'Ti6Al4V',
    quality_grade: 'A',
    laser_power: 200.0,
    scan_speed: 1000.0,
    layer_thickness: 0.03,
    density: 0.98,
    surface_roughness: 5.2
});

// Create sample material node
CREATE (m:Material {
    material_type: 'Ti6Al4V',
    properties: {
        density: 4.43,
        melting_point: 1668,
        thermal_conductivity: 7.0
    },
    supplier: 'MaterialCorp',
    certification: 'AMS4911'
});

// Create sample machine node
CREATE (machine:Machine {
    machine_id: 'MACHINE_001',
    machine_type: 'PBF-LB/M',
    model: 'EOS M290',
    status: 'operational',
    location: 'Building A, Floor 2',
    installation_date: date('2023-01-15')
});

// Create sample quality node
CREATE (q:Quality {
    grade: 'A',
    metrics: {
        density: 0.98,
        surface_roughness: 5.2,
        dimensional_accuracy: 25.0
    },
    standards: ['ISO 2768', 'ASTM F2924'],
    inspector: 'John Doe'
});

// Create sample sensor node
CREATE (s:Sensor {
    sensor_id: 'TEMP_001',
    sensor_type: 'temperature',
    location: 'build_chamber',
    model: 'PT100',
    calibration_date: date('2024-01-01'),
    accuracy: 0.1
});

// Create sample operator node
CREATE (o:Operator {
    operator_id: 'OP_001',
    name: 'Jane Smith',
    certification: 'PBF-LB/M Level 2',
    experience_years: 5,
    shift: 'day'
});

// Create sample defect node
CREATE (d:Defect {
    defect_id: 'DEF_001',
    defect_type: 'porosity',
    severity: 'minor',
    location: {x: 10.5, y: 20.3, z: 5.2},
    size: 0.1,
    detection_method: 'CT_scan'
});

// =============================================================================
// RELATIONSHIP CREATION QUERIES
// =============================================================================

// Create relationships between nodes
MATCH (p:Process {process_id: 'PROC_001'})
MATCH (m:Material {material_type: 'Ti6Al4V'})
MATCH (machine:Machine {machine_id: 'MACHINE_001'})
MATCH (q:Quality {grade: 'A'})
MATCH (s:Sensor {sensor_id: 'TEMP_001'})
MATCH (o:Operator {operator_id: 'OP_001'})
MATCH (d:Defect {defect_id: 'DEF_001'})

CREATE (p)-[:USES_MATERIAL {quantity: 1.5, unit: 'kg'}]->(m)
CREATE (p)-[:EXECUTED_ON {duration: 3600, start_time: datetime('2024-01-15T10:30:00')}]->(machine)
CREATE (p)-[:HAS_QUALITY {measured_at: datetime('2024-01-15T14:30:00')}]->(q)
CREATE (p)-[:MONITORED_BY {sampling_rate: 1.0}]->(s)
CREATE (p)-[:OPERATED_BY {shift: 'day'}]->(o)
CREATE (p)-[:HAS_DEFECT {detected_at: datetime('2024-01-15T15:00:00')}]->(d);

// =============================================================================
// ANALYTICAL QUERIES
// =============================================================================

// Query 1: Find all processes with quality grade A
MATCH (p:Process)-[:HAS_QUALITY]->(q:Quality {grade: 'A'})
RETURN p.process_id, p.material_type, q.metrics;

// Query 2: Find machines with most successful processes
MATCH (machine:Machine)<-[:EXECUTED_ON]-(p:Process)-[:HAS_QUALITY]->(q:Quality {grade: 'A'})
RETURN machine.machine_id, machine.model, count(p) as successful_processes
ORDER BY successful_processes DESC;

// Query 3: Find material types with highest quality
MATCH (m:Material)<-[:USES_MATERIAL]-(p:Process)-[:HAS_QUALITY]->(q:Quality)
RETURN m.material_type, avg(q.metrics.density) as avg_density, avg(q.metrics.surface_roughness) as avg_roughness
ORDER BY avg_density DESC;

// Query 4: Find defect patterns by material type
MATCH (m:Material)<-[:USES_MATERIAL]-(p:Process)-[:HAS_DEFECT]->(d:Defect)
RETURN m.material_type, d.defect_type, count(d) as defect_count
ORDER BY defect_count DESC;

// Query 5: Find sensor performance correlation with quality
MATCH (s:Sensor)-[:MONITORED_BY]-(p:Process)-[:HAS_QUALITY]->(q:Quality)
RETURN s.sensor_type, s.location, avg(q.metrics.density) as avg_quality
ORDER BY avg_quality DESC;

// =============================================================================
// GRAPH ALGORITHMS AND ANALYTICS
// =============================================================================

// PageRank for process importance
CALL gds.pageRank.stream('process_graph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).process_id as process_id, score
ORDER BY score DESC;

// Community detection for material groups
CALL gds.louvain.stream('material_graph')
YIELD nodeId, communityId
RETURN gds.util.asNode(nodeId).material_type as material_type, communityId
ORDER BY communityId;

// Shortest path between processes
MATCH (p1:Process {process_id: 'PROC_001'}), (p2:Process {process_id: 'PROC_002'})
CALL gds.shortestPath.dijkstra.stream('process_graph', {
    sourceNode: p1,
    targetNode: p2,
    relationshipWeightProperty: 'weight'
})
YIELD path
RETURN path;
