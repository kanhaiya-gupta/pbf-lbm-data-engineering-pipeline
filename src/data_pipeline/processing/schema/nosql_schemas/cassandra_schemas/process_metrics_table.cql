-- Cassandra table schema for process metrics time-series data
-- Keyspace: pbf_manufacturing
-- Table: process_metrics

CREATE KEYSPACE IF NOT EXISTS pbf_manufacturing
WITH REPLICATION = {
    'class': 'SimpleStrategy',
    'replication_factor': 3
};

USE pbf_manufacturing;

-- Process metrics table for time-series data
CREATE TABLE IF NOT EXISTS process_metrics (
    process_id text,
    timestamp timestamp,
    metric_type text,
    metric_value double,
    unit text,
    quality_score double,
    sensor_id text,
    machine_id text,
    material_type text,
    PRIMARY KEY ((process_id), timestamp, metric_type)
) WITH CLUSTERING ORDER BY (timestamp DESC, metric_type ASC);

-- Sensor data table for high-frequency sensor readings
CREATE TABLE IF NOT EXISTS sensor_data (
    sensor_id text,
    timestamp timestamp,
    process_id text,
    sensor_type text,
    value double,
    unit text,
    signal_quality double,
    anomaly_score double,
    PRIMARY KEY ((sensor_id), timestamp)
) WITH CLUSTERING ORDER BY (timestamp DESC);

-- Quality metrics table for aggregated quality data
CREATE TABLE IF NOT EXISTS quality_metrics (
    process_id text,
    measurement_type text,
    timestamp timestamp,
    density double,
    surface_roughness double,
    dimensional_accuracy double,
    defect_count int,
    quality_grade text,
    inspector_id text,
    PRIMARY KEY ((process_id), measurement_type, timestamp)
) WITH CLUSTERING ORDER BY (measurement_type ASC, timestamp DESC);

-- Material properties table
CREATE TABLE IF NOT EXISTS material_properties (
    material_type text,
    batch_id text,
    property_name text,
    property_value double,
    unit text,
    test_date timestamp,
    test_method text,
    PRIMARY KEY ((material_type, batch_id), property_name, test_date)
) WITH CLUSTERING ORDER BY (property_name ASC, test_date DESC);

-- Machine performance table
CREATE TABLE IF NOT EXISTS machine_performance (
    machine_id text,
    date date,
    hour int,
    total_processes int,
    successful_processes int,
    avg_quality_score double,
    avg_process_duration int,
    maintenance_events int,
    PRIMARY KEY ((machine_id), date, hour)
) WITH CLUSTERING ORDER BY (date DESC, hour DESC);

-- Defect analysis table
CREATE TABLE IF NOT EXISTS defect_analysis (
    process_id text,
    defect_id text,
    defect_type text,
    severity text,
    location_x double,
    location_y double,
    location_z double,
    size double,
    detection_method text,
    timestamp timestamp,
    PRIMARY KEY ((process_id), defect_id)
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_sensor_data_process_id ON sensor_data (process_id);
CREATE INDEX IF NOT EXISTS idx_sensor_data_sensor_type ON sensor_data (sensor_type);
CREATE INDEX IF NOT EXISTS idx_quality_metrics_quality_grade ON quality_metrics (quality_grade);
CREATE INDEX IF NOT EXISTS idx_defect_analysis_defect_type ON defect_analysis (defect_type);
CREATE INDEX IF NOT EXISTS idx_defect_analysis_severity ON defect_analysis (severity);

-- Materialized views for common queries
CREATE MATERIALIZED VIEW IF NOT EXISTS process_metrics_by_machine AS
    SELECT machine_id, process_id, timestamp, metric_type, metric_value, unit
    FROM process_metrics
    WHERE machine_id IS NOT NULL
    PRIMARY KEY ((machine_id), timestamp, process_id, metric_type);

CREATE MATERIALIZED VIEW IF NOT EXISTS sensor_data_by_type AS
    SELECT sensor_type, sensor_id, timestamp, process_id, value, unit
    FROM sensor_data
    WHERE sensor_type IS NOT NULL
    PRIMARY KEY ((sensor_type), timestamp, sensor_id);

CREATE MATERIALIZED VIEW IF NOT EXISTS quality_metrics_by_grade AS
    SELECT quality_grade, process_id, measurement_type, timestamp, density, surface_roughness
    FROM quality_metrics
    WHERE quality_grade IS NOT NULL
    PRIMARY KEY ((quality_grade), timestamp, process_id, measurement_type);

-- Data retention policies (TTL examples)
-- Note: TTL is set at insert time, not in schema
-- Example: INSERT INTO sensor_data (...) VALUES (...) USING TTL 2592000; -- 30 days
-- Example: INSERT INTO process_metrics (...) VALUES (...) USING TTL 31536000; -- 1 year

-- Compression and compaction settings
ALTER TABLE process_metrics WITH compression = {
    'class': 'LZ4Compressor',
    'chunk_length_in_kb': 64
};

ALTER TABLE sensor_data WITH compression = {
    'class': 'LZ4Compressor',
    'chunk_length_in_kb': 32
};

-- Compaction strategy for time-series data
ALTER TABLE process_metrics WITH compaction = {
    'class': 'TimeWindowCompactionStrategy',
    'compaction_window_unit': 'DAYS',
    'compaction_window_size': 1
};

ALTER TABLE sensor_data WITH compaction = {
    'class': 'TimeWindowCompactionStrategy',
    'compaction_window_unit': 'HOURS',
    'compaction_window_size': 6
};
