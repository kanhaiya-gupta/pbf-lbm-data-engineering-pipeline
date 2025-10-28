// =============================================================================
// Graph Data Science (GDS) Algorithms for PBF-LB/M Manufacturing Knowledge Graph
// Advanced graph algorithms for process optimization and quality analysis
// =============================================================================

// =============================================================================
// GRAPH PROJECTION SETUP
// =============================================================================

// Create graph projection for process analysis
CALL gds.graph.project(
    'process_graph',
    ['Process', 'Machine', 'Part', 'Build', 'Material', 'Quality', 'Sensor', 'User', 'Alert', 'Defect'],
    {
        USES_MACHINE: {orientation: 'NATURAL'},
        CREATES_PART: {orientation: 'NATURAL'},
        PART_OF_BUILD: {orientation: 'NATURAL'},
        USES_MATERIAL: {orientation: 'NATURAL'},
        HAS_QUALITY: {orientation: 'NATURAL'},
        MONITORED_BY: {orientation: 'NATURAL'},
        OPERATED_BY: {orientation: 'NATURAL'},
        GENERATES_ALERT: {orientation: 'NATURAL'},
        HOSTS_PROCESS: {orientation: 'NATURAL'},
        CREATED_BY_PROCESS: {orientation: 'NATURAL'},
        CONTAINS_PROCESS: {orientation: 'NATURAL'},
        USED_IN_PROCESS: {orientation: 'NATURAL'},
        VALIDATES_PROCESS: {orientation: 'NATURAL'},
        MONITORS_PROCESS: {orientation: 'NATURAL'},
        OPERATES_PROCESS: {orientation: 'NATURAL'},
        AFFECTS_PROCESS: {orientation: 'NATURAL'}
    },
    {
        nodeProperties: {
            Process: ['laser_power', 'scan_speed', 'layer_thickness', 'density', 'surface_roughness', 'duration', 'energy_consumption'],
            Machine: ['machine_type', 'model', 'status', 'laser_power_max', 'accuracy'],
            Part: ['part_type', 'material_type', 'volume', 'surface_area', 'weight', 'quality_grade'],
            Build: ['build_name', 'status', 'total_parts', 'success_rate', 'total_duration', 'material_usage'],
            Material: ['material_type', 'supplier', 'certification', 'condition'],
            Quality: ['grade', 'metrics', 'confidence_level'],
            Sensor: ['sensor_type', 'location', 'accuracy', 'sampling_rate', 'status'],
            User: ['role', 'department', 'active'],
            Alert: ['severity', 'status', 'threshold', 'actual_value', 'resolution_time'],
            Defect: ['defect_type', 'severity', 'size', 'confidence', 'status']
        },
        relationshipProperties: {
            USES_MACHINE: ['duration', 'start_time'],
            CREATES_PART: ['quantity', 'success_rate'],
            PART_OF_BUILD: ['sequence', 'priority'],
            USES_MATERIAL: ['quantity', 'unit'],
            HAS_QUALITY: ['measured_at'],
            MONITORED_BY: ['sampling_rate', 'active'],
            OPERATED_BY: ['shift', 'experience_level'],
            GENERATES_ALERT: ['triggered_at'],
            HOSTS_PROCESS: ['capacity', 'utilization'],
            CREATED_BY_PROCESS: ['creation_time'],
            CONTAINS_PROCESS: ['sequence', 'status'],
            USED_IN_PROCESS: ['consumption_rate'],
            VALIDATES_PROCESS: ['confidence'],
            MONITORS_PROCESS: ['coverage', 'accuracy'],
            OPERATES_PROCESS: ['supervision_level'],
            AFFECTS_PROCESS: ['impact', 'resolution']
        }
    }
) YIELD graphName, nodeCount, relationshipCount, createMillis;

// Create graph projection for machine analysis
CALL gds.graph.project(
    'machine_graph',
    ['Machine', 'Process', 'Sensor', 'Alert', 'User'],
    {
        HOSTS_PROCESS: {orientation: 'NATURAL'},
        HAS_SENSOR: {orientation: 'NATURAL'},
        OPERATED_BY: {orientation: 'NATURAL'},
        USES_MACHINE: {orientation: 'NATURAL'},
        MONITORED_BY: {orientation: 'NATURAL'},
        GENERATES_ALERT: {orientation: 'NATURAL'},
        OPERATES_MACHINE: {orientation: 'NATURAL'},
        TRIGGERS_ALERT: {orientation: 'NATURAL'}
    },
    {
        nodeProperties: {
            Machine: ['machine_type', 'model', 'status', 'location', 'laser_power_max', 'accuracy'],
            Process: ['laser_power', 'scan_speed', 'duration', 'energy_consumption', 'density', 'surface_roughness'],
            Sensor: ['sensor_type', 'location', 'accuracy', 'sampling_rate', 'status'],
            Alert: ['severity', 'status', 'threshold', 'actual_value', 'resolution_time'],
            User: ['role', 'department', 'active']
        }
    }
) YIELD graphName, nodeCount, relationshipCount, createMillis;

// Create graph projection for quality analysis
CALL gds.graph.project(
    'quality_graph',
    ['Process', 'Quality', 'Material', 'Sensor', 'Defect'],
    {
        HAS_QUALITY: {orientation: 'NATURAL'},
        USES_MATERIAL: {orientation: 'NATURAL'},
        MONITORED_BY: {orientation: 'NATURAL'},
        HAS_DEFECT: {orientation: 'NATURAL'},
        VALIDATES_PROCESS: {orientation: 'NATURAL'},
        USED_IN_PROCESS: {orientation: 'NATURAL'},
        MONITORS_PROCESS: {orientation: 'NATURAL'},
        AFFECTS_PART: {orientation: 'NATURAL'}
    },
    {
        nodeProperties: {
            Process: ['laser_power', 'scan_speed', 'layer_thickness', 'density', 'surface_roughness'],
            Quality: ['grade', 'metrics', 'confidence_level'],
            Material: ['material_type', 'supplier', 'condition'],
            Sensor: ['sensor_type', 'accuracy', 'sampling_rate'],
            Defect: ['defect_type', 'severity', 'size', 'confidence']
        }
    }
) YIELD graphName, nodeCount, relationshipCount, createMillis;

// =============================================================================
// CENTRALITY ALGORITHMS
// =============================================================================

// PageRank for Process Importance
CALL gds.pageRank.stream('process_graph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).process_id as process_id,
       gds.util.asNode(nodeId).material_type as material_type,
       gds.util.asNode(nodeId).laser_power as laser_power,
       score
ORDER BY score DESC
LIMIT 20;

// PageRank for Machine Importance
CALL gds.pageRank.stream('machine_graph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).machine_id as machine_id,
       gds.util.asNode(nodeId).machine_type as machine_type,
       gds.util.asNode(nodeId).model as model,
       score
ORDER BY score DESC
LIMIT 20;

// Betweenness Centrality for Process Flow Analysis
CALL gds.betweenness.stream('process_graph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).process_id as process_id,
       gds.util.asNode(nodeId).material_type as material_type,
       score
ORDER BY score DESC
LIMIT 20;

// Closeness Centrality for Process Accessibility
CALL gds.closeness.stream('process_graph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).process_id as process_id,
       gds.util.asNode(nodeId).material_type as material_type,
       score
ORDER BY score DESC
LIMIT 20;

// =============================================================================
// COMMUNITY DETECTION ALGORITHMS
// =============================================================================

// Louvain Community Detection for Process Groups
CALL gds.louvain.stream('process_graph')
YIELD nodeId, communityId
RETURN gds.util.asNode(nodeId).process_id as process_id,
       gds.util.asNode(nodeId).material_type as material_type,
       communityId
ORDER BY communityId, process_id;

// Label Propagation for Machine Clusters
CALL gds.labelPropagation.stream('machine_graph')
YIELD nodeId, communityId
RETURN gds.util.asNode(nodeId).machine_id as machine_id,
       gds.util.asNode(nodeId).machine_type as machine_type,
       gds.util.asNode(nodeId).model as model,
       communityId
ORDER BY communityId, machine_id;

// Weakly Connected Components for Process Networks
CALL gds.wcc.stream('process_graph')
YIELD nodeId, componentId
RETURN gds.util.asNode(nodeId).process_id as process_id,
       gds.util.asNode(nodeId).material_type as material_type,
       componentId
ORDER BY componentId, process_id;

// =============================================================================
// PATH ANALYSIS ALGORITHMS
// =============================================================================

// Shortest Path between Processes
MATCH (p1:Process {process_id: 'PROC_001'}), (p2:Process {process_id: 'PROC_002'})
CALL gds.shortestPath.dijkstra.stream('process_graph', {
    sourceNode: p1,
    targetNode: p2,
    relationshipWeightProperty: 'duration'
})
YIELD path
RETURN path;

// All Shortest Paths for Process Optimization
MATCH (p1:Process), (p2:Process)
WHERE p1.process_id < p2.process_id
CALL gds.shortestPath.dijkstra.stream('process_graph', {
    sourceNode: p1,
    targetNode: p2,
    relationshipWeightProperty: 'duration'
})
YIELD path
RETURN p1.process_id as source_process,
       p2.process_id as target_process,
       length(path) as path_length
ORDER BY path_length
LIMIT 10;

// =============================================================================
// SIMILARITY ALGORITHMS
// =============================================================================

// Node Similarity for Process Comparison
CALL gds.nodeSimilarity.stream('process_graph')
YIELD node1, node2, similarity
RETURN gds.util.asNode(node1).process_id as process1,
       gds.util.asNode(node2).process_id as process2,
       similarity
ORDER BY similarity DESC
LIMIT 20;

// Jaccard Similarity for Machine Comparison
CALL gds.nodeSimilarity.jaccard.stream('machine_graph')
YIELD node1, node2, similarity
RETURN gds.util.asNode(node1).machine_id as machine1,
       gds.util.asNode(node2).machine_id as machine2,
       similarity
ORDER BY similarity DESC
LIMIT 20;

// =============================================================================
// LINK PREDICTION ALGORITHMS
// =============================================================================

// Adamic-Adar for Relationship Prediction
CALL gds.adamicAdar.stream('process_graph')
YIELD node1, node2, score
RETURN gds.util.asNode(node1).process_id as process1,
       gds.util.asNode(node2).process_id as process2,
       score
ORDER BY score DESC
LIMIT 20;

// Common Neighbors for Process Relationships
CALL gds.commonNeighbors.stream('process_graph')
YIELD node1, node2, score
RETURN gds.util.asNode(node1).process_id as process1,
       gds.util.asNode(node2).process_id as process2,
       score
ORDER BY score DESC
LIMIT 20;

// =============================================================================
// EMBEDDING ALGORITHMS
// =============================================================================

// Fast Random Projection for Process Embeddings
CALL gds.fastRP.stream('process_graph', {
    embeddingDimension: 64,
    iterationWeights: [0.8, 1.0, 1.0],
    normalizationStrength: 0.05,
    randomSeed: 42
})
YIELD nodeId, embedding
RETURN gds.util.asNode(node1).process_id as process_id,
       embedding
ORDER BY process_id
LIMIT 20;

// GraphSAGE for Machine Embeddings
CALL gds.beta.graphSage.stream('machine_graph', {
    embeddingDimension: 64,
    aggregator: 'mean',
    activationFunction: 'sigmoid',
    randomSeed: 42
})
YIELD nodeId, embedding
RETURN gds.util.asNode(nodeId).machine_id as machine_id,
       embedding
ORDER BY machine_id
LIMIT 20;

// =============================================================================
// CLUSTERING ALGORITHMS
// =============================================================================

// K-Means Clustering for Process Groups
CALL gds.kmeans.stream('process_graph', {
    nodeProperties: ['laser_power', 'scan_speed', 'layer_thickness', 'density'],
    k: 5,
    randomSeed: 42
})
YIELD nodeId, communityId
RETURN gds.util.asNode(nodeId).process_id as process_id,
       gds.util.asNode(nodeId).material_type as material_type,
       communityId
ORDER BY communityId, process_id;

// =============================================================================
// ANOMALY DETECTION ALGORITHMS
// =============================================================================

// Local Outlier Factor for Anomaly Detection
CALL gds.localOutlierFactor.stream('process_graph', {
    nodeProperties: ['laser_power', 'scan_speed', 'density', 'surface_roughness'],
    k: 5
})
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).process_id as process_id,
       gds.util.asNode(nodeId).material_type as material_type,
       score
ORDER BY score DESC
LIMIT 20;

// =============================================================================
// GRAPH ALGORITHM METRICS
// =============================================================================

// Graph Statistics
CALL gds.graph.stats('process_graph')
YIELD nodeCount, relationshipCount, density, averageDegree;

// Algorithm Performance Metrics
CALL gds.pageRank.stats('process_graph')
YIELD computeMillis, nodeCount, relationshipCount;

// =============================================================================
// CUSTOM ALGORITHMS FOR MANUFACTURING
// =============================================================================

// Process Quality Prediction Algorithm
MATCH (p:Process)-[:HAS_QUALITY]->(q:Quality)
WITH p, q
MATCH (p)-[:USES_MACHINE]->(m:Machine)
WITH p, q, m
MATCH (p)-[:USES_MATERIAL]->(mat:Material)
WITH p, q, m, mat
MATCH (p)-[:MONITORED_BY]->(s:Sensor)
RETURN p.process_id as process_id,
       p.laser_power as laser_power,
       p.scan_speed as scan_speed,
       p.layer_thickness as layer_thickness,
       m.machine_type as machine_type,
       mat.material_type as material_type,
       s.sensor_type as sensor_type,
       q.grade as quality_grade,
       q.metrics.density as density,
       q.metrics.surface_roughness as surface_roughness
ORDER BY q.grade DESC, q.metrics.density DESC;

// Machine Performance Ranking Algorithm
MATCH (m:Machine)<-[:USES_MACHINE]-(p:Process)-[:HAS_QUALITY]->(q:Quality)
WITH m, count(p) as total_processes, 
     sum(CASE WHEN q.grade = 'A' THEN 1 ELSE 0 END) as successful_processes,
     avg(p.duration) as avg_duration,
     avg(p.energy_consumption) as avg_energy,
     avg(q.metrics.density) as avg_density
RETURN m.machine_id as machine_id,
       m.machine_type as machine_type,
       m.model as model,
       total_processes,
       successful_processes,
       successful_processes * 100.0 / total_processes as success_rate,
       avg_duration,
       avg_energy,
       avg_density
ORDER BY success_rate DESC, avg_density DESC;

// Process Optimization Recommendation Algorithm
MATCH (p:Process)-[:HAS_QUALITY]->(q:Quality)
WHERE q.grade = 'A'
WITH p, q
MATCH (p)-[:USES_MACHINE]->(m:Machine)
WITH p, q, m
MATCH (p)-[:USES_MATERIAL]->(mat:Material)
WITH p, q, m, mat
MATCH (p)-[:MONITORED_BY]->(s:Sensor)
RETURN p.material_type as material_type,
       m.machine_type as machine_type,
       mat.supplier as material_supplier,
       s.sensor_type as sensor_type,
       avg(p.laser_power) as optimal_laser_power,
       avg(p.scan_speed) as optimal_scan_speed,
       avg(p.layer_thickness) as optimal_layer_thickness,
       avg(q.metrics.density) as target_density,
       avg(q.metrics.surface_roughness) as target_roughness,
       count(p) as sample_size
ORDER BY sample_size DESC;

// =============================================================================
// GRAPH CLEANUP
// =============================================================================

// Drop graph projections when done
CALL gds.graph.drop('process_graph');
CALL gds.graph.drop('machine_graph');
CALL gds.graph.drop('quality_graph');
