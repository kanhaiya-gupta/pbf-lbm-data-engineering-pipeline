-- ClickHouse Schema for Cassandra Time Series Data
-- Optimized for time-series analytics and sensor data

CREATE TABLE IF NOT EXISTS cassandra_time_series (
    -- Primary identifiers
    id UInt64,
    sensor_id String,
    process_id Nullable(String),
    build_id Nullable(String),
    machine_id Nullable(String),
    timestamp DateTime,
    
    -- Time series data
    sensor_type Nullable(String),
    value Nullable(Float64),
    unit Nullable(String),
    location Nullable(String),
    status Nullable(String),
    quality_score Nullable(Float64),
    
    -- Aggregation data
    aggregation_type Nullable(String),
    aggregation_window Nullable(String),
    min_value Nullable(Float64),
    max_value Nullable(Float64),
    avg_value Nullable(Float64),
    sum_value Nullable(Float64),
    count_value Nullable(UInt32),
    std_dev Nullable(Float64),
    
    -- Calibration data
    calibration_factor Nullable(Float64),
    calibration_accuracy Nullable(Float64),
    calibration_uncertainty Nullable(Float64),
    last_calibrated Nullable(DateTime),
    
    -- Metadata
    created_at DateTime,
    
    -- Indexes for performance
    INDEX idx_sensor_id sensor_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_process_id process_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_sensor_type sensor_type TYPE bloom_filter GRANULARITY 1,
    INDEX idx_aggregation_type aggregation_type TYPE bloom_filter GRANULARITY 1,
    INDEX idx_timestamp timestamp TYPE minmax GRANULARITY 1
) 
ENGINE = MergeTree()
ORDER BY (timestamp, sensor_id)
PARTITION BY toYYYYMM(timestamp)
TTL timestamp + INTERVAL 6 MONTH
SETTINGS index_granularity = 8192;

-- Materialized view for time series analytics
CREATE MATERIALIZED VIEW IF NOT EXISTS time_series_analytics
ENGINE = SummingMergeTree()
ORDER BY (sensor_type, aggregation_type, minute)
AS SELECT
    sensor_type,
    aggregation_type,
    toStartOfMinute(timestamp) as minute,
    COUNT(*) as record_count,
    AVG(value) as avg_value,
    MIN(value) as min_value,
    MAX(value) as max_value,
    stddevPop(value) as std_dev,
    quantile(0.95)(value) as p95_value,
    AVG(quality_score) as avg_quality
FROM cassandra_time_series
GROUP BY sensor_type, aggregation_type, toStartOfMinute(timestamp);
