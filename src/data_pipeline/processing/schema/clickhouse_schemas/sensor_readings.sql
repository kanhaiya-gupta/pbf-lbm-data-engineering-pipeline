-- ClickHouse Schema for Sensor Readings Data
-- Optimized for time-series analytics and real-time monitoring

CREATE TABLE IF NOT EXISTS sensor_readings (
    -- Primary identifiers
    id UInt64,
    sensor_id String,
    process_id Nullable(String),
    build_id Nullable(String),
    machine_id Nullable(String),
    timestamp DateTime,
    
    -- Sensor information
    sensor_type Nullable(String),
    value Nullable(Float64),
    unit Nullable(String),
    location Nullable(String),
    status Nullable(String),
    quality_score Nullable(Float64),
    
    -- Calibration data
    calibration_date Nullable(DateTime),
    calibration_factor Nullable(Float64),
    calibration_accuracy Nullable(Float64),
    calibration_uncertainty Nullable(Float64),
    
    -- Measurement metadata
    sampling_rate Nullable(UInt32),
    data_duration Nullable(Float64),
    data_points Nullable(UInt32),
    min_value Nullable(Float64),
    max_value Nullable(Float64),
    
    -- Processing status
    processing_status Nullable(String),
    data_quality Nullable(String),
    file_path Nullable(String),
    file_size Nullable(UInt64),
    file_hash Nullable(String),
    
    -- Metadata
    created_at DateTime,
    
    -- Indexes for performance
    INDEX idx_sensor_id sensor_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_process_id process_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_sensor_type sensor_type TYPE bloom_filter GRANULARITY 1,
    INDEX idx_timestamp timestamp TYPE minmax GRANULARITY 1
) 
ENGINE = MergeTree()
ORDER BY (timestamp, sensor_id)
PARTITION BY toYYYYMM(timestamp)
TTL timestamp + INTERVAL 6 MONTH
SETTINGS index_granularity = 8192;

-- Materialized view for sensor analytics
CREATE MATERIALIZED VIEW IF NOT EXISTS sensor_analytics
ENGINE = SummingMergeTree()
ORDER BY (sensor_type, minute)
AS SELECT
    sensor_type,
    toStartOfMinute(timestamp) as minute,
    COUNT(*) as reading_count,
    AVG(value) as avg_value,
    MIN(value) as min_value,
    MAX(value) as max_value,
    stddevPop(value) as std_dev,
    quantile(0.95)(value) as p95_value,
    AVG(quality_score) as avg_quality
FROM sensor_readings
GROUP BY sensor_type, toStartOfMinute(timestamp);

-- Materialized view for sensor health monitoring
CREATE MATERIALIZED VIEW IF NOT EXISTS sensor_health_monitoring
ENGINE = SummingMergeTree()
ORDER BY (sensor_id, date)
AS SELECT
    sensor_id,
    toStartOfDay(timestamp) as date,
    COUNT(*) as total_readings,
    AVG(quality_score) as avg_quality,
    COUNTIf(status = 'error') as error_count,
    COUNTIf(data_quality = 'poor') as poor_quality_count,
    AVG(calibration_accuracy) as avg_calibration_accuracy
FROM sensor_readings
GROUP BY sensor_id, toStartOfDay(timestamp);
