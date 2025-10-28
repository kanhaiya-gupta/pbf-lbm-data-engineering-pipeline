-- ClickHouse Schema for Raw Sensor Data (from MongoDB)
-- Optimized for time-series sensor data and file metadata

CREATE TABLE IF NOT EXISTS raw_sensor_data (
    -- Primary identifiers
    id UInt64,
    sensor_id String,
    process_id Nullable(String),
    build_id Nullable(String),
    machine_id Nullable(String),
    timestamp DateTime,
    
    -- File information
    file_path Nullable(String),
    file_size Nullable(UInt64),
    file_hash Nullable(String),
    file_status Nullable(String),
    
    -- Sensor metadata
    sensor_type Nullable(String),
    data_format Nullable(String),
    sampling_rate Nullable(UInt32),
    data_duration Nullable(Float64),
    data_points Nullable(UInt32),
    processing_status Nullable(String),
    data_quality Nullable(String),
    
    -- Calibration information
    last_calibrated Nullable(Date),
    calibration_factor Nullable(Float64),
    calibration_accuracy Nullable(Float64),
    calibration_uncertainty Nullable(Float64),
    
    -- Measurement range
    min_value Nullable(Float64),
    max_value Nullable(Float64),
    unit Nullable(String),
    
    -- Processing metadata
    processing_algorithm Nullable(String),
    processing_parameters Nullable(String),
    processing_version Nullable(String),
    processing_timestamp Nullable(DateTime),
    
    -- Quality metrics
    signal_to_noise_ratio Nullable(Float64),
    data_completeness Nullable(Float64),
    outlier_percentage Nullable(Float64),
    
    -- Metadata
    created_at DateTime,
    
    -- Indexes for performance
    INDEX idx_sensor_id sensor_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_process_id process_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_sensor_type sensor_type TYPE bloom_filter GRANULARITY 1,
    INDEX idx_file_status file_status TYPE bloom_filter GRANULARITY 1,
    INDEX idx_timestamp timestamp TYPE minmax GRANULARITY 1
) 
ENGINE = MergeTree()
ORDER BY (timestamp, sensor_id)
PARTITION BY toYYYYMM(timestamp)
TTL timestamp + INTERVAL 6 MONTH
SETTINGS index_granularity = 8192;

-- Materialized view for sensor data analytics
CREATE MATERIALIZED VIEW IF NOT EXISTS sensor_data_analytics
ENGINE = SummingMergeTree()
ORDER BY (sensor_type, date)
AS SELECT
    sensor_type,
    toStartOfDay(timestamp) as date,
    COUNT(*) as file_count,
    SUM(file_size) as total_file_size,
    AVG(data_duration) as avg_duration,
    AVG(data_points) as avg_data_points,
    AVG(signal_to_noise_ratio) as avg_snr,
    AVG(data_completeness) as avg_completeness,
    COUNTIf(data_quality = 'excellent') as excellent_quality_count,
    COUNTIf(data_quality = 'poor') as poor_quality_count
FROM raw_sensor_data
GROUP BY sensor_type, toStartOfDay(timestamp);

-- Materialized view for processing status tracking
CREATE MATERIALIZED VIEW IF NOT EXISTS processing_status_tracking
ENGINE = SummingMergeTree()
ORDER BY (processing_status, hour)
AS SELECT
    processing_status,
    toStartOfHour(timestamp) as hour,
    COUNT(*) as file_count,
    SUM(file_size) as total_size,
    AVG(processing_timestamp - timestamp) as avg_processing_delay,
    uniq(sensor_type) as unique_sensor_types,
    uniq(machine_id) as unique_machines
FROM raw_sensor_data
GROUP BY processing_status, toStartOfHour(timestamp);
