-- ClickHouse Schema for ISPM Monitoring Data
-- Optimized for industrial monitoring and anomaly detection

CREATE TABLE IF NOT EXISTS ispm_monitoring (
    -- Primary identifiers
    id UInt64,
    monitoring_id String,
    process_id Nullable(String),
    build_id Nullable(String),
    machine_id Nullable(String),
    timestamp DateTime,
    
    -- Monitoring data
    monitoring_type Nullable(String),
    sensor_id Nullable(String),
    value Nullable(Float64),
    unit Nullable(String),
    location Nullable(String),
    status Nullable(String),
    
    -- Anomaly detection
    anomaly_score Nullable(Float64),
    anomaly_type Nullable(String),
    anomaly_severity Nullable(String),
    anomaly_confidence Nullable(Float64),
    anomaly_detection_method Nullable(String),
    
    -- Quality metrics
    quality_score Nullable(Float64),
    data_quality Nullable(String),
    measurement_uncertainty Nullable(Float64),
    calibration_status Nullable(String),
    
    -- Environmental conditions
    temperature Nullable(Float64),
    humidity Nullable(Float64),
    pressure Nullable(Float64),
    vibration Nullable(Float64),
    noise_level Nullable(Float64),
    
    -- Process parameters
    process_speed Nullable(Float64),
    process_pressure Nullable(Float64),
    process_temperature Nullable(Float64),
    process_flow_rate Nullable(Float64),
    
    -- Metadata
    created_at DateTime,
    
    -- Indexes for performance
    INDEX idx_monitoring_id monitoring_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_process_id process_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_monitoring_type monitoring_type TYPE bloom_filter GRANULARITY 1,
    INDEX idx_anomaly_type anomaly_type TYPE bloom_filter GRANULARITY 1,
    INDEX idx_timestamp timestamp TYPE minmax GRANULARITY 1
) 
ENGINE = MergeTree()
ORDER BY (timestamp, monitoring_id)
PARTITION BY toYYYYMM(timestamp)
TTL timestamp + INTERVAL 1 YEAR
SETTINGS index_granularity = 8192;

-- Materialized view for anomaly detection analytics
CREATE MATERIALIZED VIEW IF NOT EXISTS anomaly_detection_analytics
ENGINE = SummingMergeTree()
ORDER BY (anomaly_type, anomaly_severity, date)
AS SELECT
    anomaly_type,
    anomaly_severity,
    toStartOfDay(timestamp) as date,
    COUNT(*) as anomaly_count,
    AVG(anomaly_score) as avg_anomaly_score,
    AVG(anomaly_confidence) as avg_confidence,
    AVG(quality_score) as avg_quality_during_anomaly,
    uniq(process_id) as affected_processes,
    uniq(machine_id) as affected_machines
FROM ispm_monitoring
WHERE anomaly_score > 0
GROUP BY anomaly_type, anomaly_severity, toStartOfDay(timestamp);
