-- ClickHouse Schema for Process Images Data (from MongoDB)
-- Optimized for process monitoring and image analytics

CREATE TABLE IF NOT EXISTS process_images (
    -- Primary identifiers
    id UInt64,
    image_id String,
    process_id Nullable(String),
    build_id Nullable(String),
    part_id Nullable(String),
    machine_id Nullable(String),
    timestamp DateTime,
    
    -- File information
    file_path Nullable(String),
    file_name Nullable(String),
    file_size Nullable(UInt64),
    file_hash Nullable(String),
    file_format Nullable(String),
    file_status Nullable(String),
    
    -- Image metadata
    image_type Nullable(String),
    image_width Nullable(UInt32),
    image_height Nullable(UInt32),
    image_depth Nullable(UInt8),
    image_channels Nullable(UInt8),
    image_resolution Nullable(Float64),
    image_compression Nullable(String),
    
    -- Process information
    process_stage Nullable(String),
    layer_number Nullable(UInt32),
    process_temperature Nullable(Float64),
    process_pressure Nullable(Float64),
    process_speed Nullable(Float64),
    laser_power Nullable(Float64),
    
    -- Image processing
    processing_status Nullable(String),
    processing_algorithm Nullable(String),
    processing_parameters Nullable(String),
    processing_duration Nullable(Float64),
    processing_timestamp Nullable(DateTime),
    
    -- Quality metrics
    image_quality_score Nullable(Float64),
    noise_level Nullable(Float64),
    contrast_ratio Nullable(Float64),
    sharpness_score Nullable(Float64),
    brightness Nullable(Float64),
    exposure_level Nullable(Float64),
    
    -- Defect analysis
    defect_count Nullable(UInt32),
    defect_types Array(String),
    defect_severity Nullable(String),
    defect_locations Array(String),
    defect_areas Array(Float64),
    
    -- Process analysis
    process_quality_score Nullable(Float64),
    layer_completeness Nullable(Float64),
    dimensional_accuracy Nullable(Float64),
    surface_quality Nullable(String),
    
    -- User and session data
    user_id Nullable(String),
    session_id Nullable(String),
    operator_id Nullable(String),
    
    -- Metadata
    created_at DateTime,
    
    -- Indexes for performance
    INDEX idx_image_id image_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_process_id process_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_image_type image_type TYPE bloom_filter GRANULARITY 1,
    INDEX idx_process_stage process_stage TYPE bloom_filter GRANULARITY 1,
    INDEX idx_timestamp timestamp TYPE minmax GRANULARITY 1
) 
ENGINE = MergeTree()
ORDER BY (timestamp, image_id)
PARTITION BY toYYYYMM(timestamp)
TTL timestamp + INTERVAL 2 YEAR
SETTINGS index_granularity = 8192;

-- Materialized view for process image analytics
CREATE MATERIALIZED VIEW IF NOT EXISTS process_image_analytics
ENGINE = SummingMergeTree()
ORDER BY (process_stage, date)
AS SELECT
    process_stage,
    toStartOfDay(timestamp) as date,
    COUNT(*) as image_count,
    AVG(file_size) as avg_file_size,
    AVG(image_quality_score) as avg_quality,
    AVG(process_quality_score) as avg_process_quality,
    AVG(layer_completeness) as avg_completeness,
    SUM(defect_count) as total_defects,
    COUNTIf(defect_severity = 'critical') as critical_defects,
    uniq(process_id) as unique_processes
FROM process_images
GROUP BY process_stage, toStartOfDay(timestamp);
