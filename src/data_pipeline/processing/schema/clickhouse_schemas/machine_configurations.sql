-- ClickHouse Schema for Machine Configurations Data (from MongoDB)
-- Optimized for configuration management and calibration tracking

CREATE TABLE IF NOT EXISTS machine_configurations (
    -- Primary identifiers
    id UInt64,
    config_id String,
    process_id Nullable(String),
    build_id Nullable(String),
    machine_id Nullable(String),
    timestamp DateTime,
    
    -- Configuration metadata
    config_name Nullable(String),
    config_type Nullable(String),
    config_version Nullable(String),
    config_format Nullable(String),
    
    -- File information
    config_data_path Nullable(String),
    config_file_size Nullable(UInt64),
    config_data_hash Nullable(String),
    
    -- Laser settings
    laser_power Nullable(UInt32),
    laser_speed Nullable(UInt32),
    laser_frequency Nullable(UInt32),
    laser_power_calibrated Nullable(Float64),
    laser_calibration_date Nullable(DateTime),
    laser_calibration_accuracy Nullable(Float64),
    
    -- Temperature settings
    bed_temperature Nullable(Float64),
    chamber_temperature Nullable(Float64),
    max_temperature Nullable(Float64),
    max_pressure Nullable(Float64),
    
    -- Powder settings
    layer_thickness Nullable(Float64),
    powder_density Nullable(Float64),
    
    -- Safety and limits
    safety_limits_max_temp Nullable(Float64),
    safety_limits_max_pressure Nullable(Float64),
    
    -- Calibration data
    calibration_date Nullable(DateTime),
    calibration_accuracy Nullable(Float64),
    calibration_uncertainty Nullable(Float64),
    calibration_factor Nullable(Float64),
    
    -- Configuration status
    config_status Nullable(String),
    is_active Nullable(UInt8),
    config_priority Nullable(UInt32),
    
    -- Metadata
    created_at DateTime,
    
    -- Indexes for performance
    INDEX idx_config_id config_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_machine_id machine_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_config_type config_type TYPE bloom_filter GRANULARITY 1,
    INDEX idx_config_status config_status TYPE bloom_filter GRANULARITY 1,
    INDEX idx_timestamp timestamp TYPE minmax GRANULARITY 1
) 
ENGINE = MergeTree()
ORDER BY (timestamp, config_id)
PARTITION BY toYYYYMM(timestamp)
TTL timestamp + INTERVAL 2 YEAR
SETTINGS index_granularity = 8192;

-- Materialized view for configuration analytics
CREATE MATERIALIZED VIEW IF NOT EXISTS configuration_analytics
ENGINE = SummingMergeTree()
ORDER BY (config_type, machine_id, date)
AS SELECT
    config_type,
    machine_id,
    toStartOfDay(timestamp) as date,
    COUNT(*) as config_count,
    AVG(laser_calibration_accuracy) as avg_laser_accuracy,
    AVG(calibration_accuracy) as avg_calibration_accuracy,
    COUNTIf(is_active = 1) as active_configs,
    uniq(process_id) as unique_processes
FROM machine_configurations
GROUP BY config_type, machine_id, toStartOfDay(timestamp);

-- Materialized view for calibration tracking
CREATE MATERIALIZED VIEW IF NOT EXISTS calibration_tracking
ENGINE = SummingMergeTree()
ORDER BY (machine_id, month)
AS SELECT
    machine_id,
    toStartOfMonth(timestamp) as month,
    COUNT(*) as calibration_count,
    AVG(laser_calibration_accuracy) as avg_laser_accuracy,
    AVG(calibration_accuracy) as avg_calibration_accuracy,
    MIN(calibration_date) as earliest_calibration,
    MAX(calibration_date) as latest_calibration
FROM machine_configurations
WHERE calibration_date IS NOT NULL
GROUP BY machine_id, toStartOfMonth(timestamp);
