-- ClickHouse Schema for Machine Build Files Data (from MongoDB)
-- Optimized for machine build file management and analytics

CREATE TABLE IF NOT EXISTS machine_build_files (
    -- Primary identifiers
    id UInt64,
    file_id String,
    machine_id Nullable(String),
    build_id Nullable(String),
    process_id Nullable(String),
    timestamp DateTime,
    
    -- File information
    file_path Nullable(String),
    file_name Nullable(String),
    file_size Nullable(UInt64),
    file_hash Nullable(String),
    file_format Nullable(String),
    file_status Nullable(String),
    
    -- Build file metadata
    file_type Nullable(String),
    file_category Nullable(String),
    file_version Nullable(String),
    file_priority Nullable(UInt32),
    file_dependencies Array(String),
    
    -- Machine information
    machine_type Nullable(String),
    machine_model Nullable(String),
    machine_serial_number Nullable(String),
    machine_firmware_version Nullable(String),
    machine_software_version Nullable(String),
    
    -- Build information
    build_type Nullable(String),
    build_parameters Nullable(String),
    build_configuration Nullable(String),
    build_environment Nullable(String),
    
    -- File processing
    processing_status Nullable(String),
    processing_algorithm Nullable(String),
    processing_parameters Nullable(String),
    processing_duration Nullable(Float64),
    processing_timestamp Nullable(DateTime),
    
    -- Quality metrics
    file_quality_score Nullable(Float64),
    file_integrity_score Nullable(Float64),
    file_completeness Nullable(Float64),
    file_validation_status Nullable(String),
    
    -- Usage analytics
    usage_count Nullable(UInt32),
    last_accessed Nullable(DateTime),
    access_frequency Nullable(Float64),
    user_rating Nullable(Float64),
    
    -- User and session data
    user_id Nullable(String),
    session_id Nullable(String),
    operator_id Nullable(String),
    
    -- Metadata
    created_at DateTime,
    
    -- Indexes for performance
    INDEX idx_file_id file_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_machine_id machine_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_file_type file_type TYPE bloom_filter GRANULARITY 1,
    INDEX idx_file_category file_category TYPE bloom_filter GRANULARITY 1,
    INDEX idx_timestamp timestamp TYPE minmax GRANULARITY 1
) 
ENGINE = MergeTree()
ORDER BY (timestamp, file_id)
PARTITION BY toYYYYMM(timestamp)
TTL timestamp + INTERVAL 1 YEAR
SETTINGS index_granularity = 8192;

-- Materialized view for machine build file analytics
CREATE MATERIALIZED VIEW IF NOT EXISTS machine_build_file_analytics
ENGINE = SummingMergeTree()
ORDER BY (file_type, machine_type, date)
AS SELECT
    file_type,
    machine_type,
    toStartOfDay(timestamp) as date,
    COUNT(*) as file_count,
    AVG(file_size) as avg_file_size,
    AVG(file_quality_score) as avg_quality,
    AVG(usage_count) as avg_usage,
    AVG(user_rating) as avg_rating,
    uniq(machine_id) as unique_machines,
    uniq(build_id) as unique_builds
FROM machine_build_files
GROUP BY file_type, machine_type, toStartOfDay(timestamp);
