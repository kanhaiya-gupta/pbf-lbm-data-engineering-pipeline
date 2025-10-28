-- Sensor Readings Time-Series Table
-- Optimized for high-frequency sensor data storage and retrieval

CREATE TABLE IF NOT EXISTS sensor_readings (
    -- Partition key: sensor_id for efficient sensor-based queries
    sensor_id text,
    
    -- Clustering key: timestamp for time-series ordering
    timestamp timestamp,
    
    -- Process and machine context
    process_id text,
    machine_id text,
    
    -- Sensor data
    sensor_type text,
    value double,
    unit text,
    quality_score double,
    
    -- Metadata
    location text,
    calibration_date timestamp,
    metadata text, -- JSON string for Dict[str, Any]
    
    -- Record metadata
    created_at timestamp,
    
    -- Primary key definition
    PRIMARY KEY (sensor_id, timestamp)
) WITH CLUSTERING ORDER BY (timestamp DESC);

-- Create indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_sensor_readings_process_id ON sensor_readings (process_id);
CREATE INDEX IF NOT EXISTS idx_sensor_readings_machine_id ON sensor_readings (machine_id);
CREATE INDEX IF NOT EXISTS idx_sensor_readings_sensor_type ON sensor_readings (sensor_type);
CREATE INDEX IF NOT EXISTS idx_sensor_readings_quality_score ON sensor_readings (quality_score);

-- Create materialized view for process-based queries
CREATE MATERIALIZED VIEW IF NOT EXISTS sensor_readings_by_process AS
    SELECT process_id, timestamp, sensor_id, machine_id, sensor_type, value, unit, quality_score
    FROM sensor_readings
    WHERE process_id IS NOT NULL AND timestamp IS NOT NULL AND sensor_id IS NOT NULL
    PRIMARY KEY (process_id, timestamp, sensor_id)
    WITH CLUSTERING ORDER BY (timestamp DESC);

-- Create materialized view for machine-based queries
CREATE MATERIALIZED VIEW IF NOT EXISTS sensor_readings_by_machine AS
    SELECT machine_id, timestamp, sensor_id, process_id, sensor_type, value, unit, quality_score
    FROM sensor_readings
    WHERE machine_id IS NOT NULL AND timestamp IS NOT NULL AND sensor_id IS NOT NULL
    PRIMARY KEY (machine_id, timestamp, sensor_id)
    WITH CLUSTERING ORDER BY (timestamp DESC);

-- Create materialized view for sensor type queries
CREATE MATERIALIZED VIEW IF NOT EXISTS sensor_readings_by_type AS
    SELECT sensor_type, timestamp, sensor_id, process_id, machine_id, value, unit, quality_score
    FROM sensor_readings
    WHERE sensor_type IS NOT NULL AND timestamp IS NOT NULL AND sensor_id IS NOT NULL
    PRIMARY KEY (sensor_type, timestamp, sensor_id)
    WITH CLUSTERING ORDER BY (timestamp DESC);

-- Set table properties for time-series optimization
ALTER TABLE sensor_readings WITH
    compaction = {'class': 'TimeWindowCompactionStrategy', 'compaction_window_unit': 'DAYS', 'compaction_window_size': 1}
    AND compression = {'sstable_compression': 'LZ4Compressor'}
    AND gc_grace_seconds = 864000; -- 10 days

-- Create TTL for automatic data retention (1 year)
-- Note: TTL is set per record during insertion, not at table level
