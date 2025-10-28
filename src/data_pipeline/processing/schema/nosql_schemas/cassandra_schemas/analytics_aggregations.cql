-- Analytics Aggregations Table
-- Optimized for pre-computed analytics and metrics

CREATE TABLE IF NOT EXISTS analytics_aggregations (
    -- Partition key: aggregation_id for efficient aggregation-based queries
    aggregation_id text,
    
    -- Clustering key: timestamp for time-series ordering
    timestamp timestamp,
    
    -- Context identifiers
    process_id text,
    machine_id text,
    sensor_id text,
    
    -- Aggregation metadata
    aggregation_type text,
    time_window text,
    granularity text,
    
    -- Aggregated metrics
    count int,
    min_value double,
    max_value double,
    avg_value double,
    median_value double,
    std_dev double,
    percentiles text, -- JSON string for Dict[str, float]
    
    -- Quality metrics
    data_quality_score double,
    missing_data_points int,
    
    -- Calculation metadata
    calculation_method text,
    metadata text, -- JSON string for Dict[str, Any]
    
    -- Record metadata
    created_at timestamp,
    
    -- Primary key definition
    PRIMARY KEY (aggregation_id, timestamp)
) WITH CLUSTERING ORDER BY (timestamp DESC);

-- Create indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_analytics_aggregations_process_id ON analytics_aggregations (process_id);
CREATE INDEX IF NOT EXISTS idx_analytics_aggregations_machine_id ON analytics_aggregations (machine_id);
CREATE INDEX IF NOT EXISTS idx_analytics_aggregations_sensor_id ON analytics_aggregations (sensor_id);
CREATE INDEX IF NOT EXISTS idx_analytics_aggregations_type ON analytics_aggregations (aggregation_type);
CREATE INDEX IF NOT EXISTS idx_analytics_aggregations_time_window ON analytics_aggregations (time_window);

-- Create materialized view for process-based queries
CREATE MATERIALIZED VIEW IF NOT EXISTS analytics_aggregations_by_process AS
    SELECT process_id, timestamp, aggregation_id, aggregation_type, avg_value, data_quality_score
    FROM analytics_aggregations
    WHERE process_id IS NOT NULL AND timestamp IS NOT NULL AND aggregation_id IS NOT NULL
    PRIMARY KEY (process_id, timestamp, aggregation_id)
    WITH CLUSTERING ORDER BY (timestamp DESC);

-- Create materialized view for machine-based queries
CREATE MATERIALIZED VIEW IF NOT EXISTS analytics_aggregations_by_machine AS
    SELECT machine_id, timestamp, aggregation_id, aggregation_type, avg_value, data_quality_score
    FROM analytics_aggregations
    WHERE machine_id IS NOT NULL AND timestamp IS NOT NULL AND aggregation_id IS NOT NULL
    PRIMARY KEY (machine_id, timestamp, aggregation_id)
    WITH CLUSTERING ORDER BY (timestamp DESC);

-- Create materialized view for sensor-based queries
CREATE MATERIALIZED VIEW IF NOT EXISTS analytics_aggregations_by_sensor AS
    SELECT sensor_id, timestamp, aggregation_id, aggregation_type, avg_value, data_quality_score
    FROM analytics_aggregations
    WHERE sensor_id IS NOT NULL AND timestamp IS NOT NULL AND aggregation_id IS NOT NULL
    PRIMARY KEY (sensor_id, timestamp, aggregation_id)
    WITH CLUSTERING ORDER BY (timestamp DESC);

-- Create materialized view for aggregation type queries
CREATE MATERIALIZED VIEW IF NOT EXISTS analytics_aggregations_by_type AS
    SELECT aggregation_type, timestamp, aggregation_id, process_id, machine_id, avg_value
    FROM analytics_aggregations
    WHERE aggregation_type IS NOT NULL AND timestamp IS NOT NULL AND aggregation_id IS NOT NULL
    PRIMARY KEY (aggregation_type, timestamp, aggregation_id)
    WITH CLUSTERING ORDER BY (timestamp DESC);

-- Set table properties for time-series optimization
ALTER TABLE analytics_aggregations WITH
    compaction = {'class': 'TimeWindowCompactionStrategy', 'compaction_window_unit': 'DAYS', 'compaction_window_size': 1}
    AND compression = {'sstable_compression': 'LZ4Compressor'}
    AND gc_grace_seconds = 864000; -- 10 days
