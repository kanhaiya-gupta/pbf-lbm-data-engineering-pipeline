-- ClickHouse Schema for CT Scan Images Data (from MongoDB)
-- Optimized for medical imaging and defect analysis

CREATE TABLE IF NOT EXISTS ct_scan_images (
    -- Primary identifiers
    id UInt64,
    scan_id String,
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
    
    -- CT scan metadata
    scan_type Nullable(String),
    scan_resolution_x Nullable(UInt32),
    scan_resolution_y Nullable(UInt32),
    scan_resolution_z Nullable(UInt32),
    voxel_size_x Nullable(Float64),
    voxel_size_y Nullable(Float64),
    voxel_size_z Nullable(Float64),
    scan_duration Nullable(Float64),
    scan_parameters Nullable(String),
    
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
    artifact_count Nullable(UInt32),
    
    -- Defect analysis
    defect_count Nullable(UInt32),
    defect_types Array(String),
    defect_severity Nullable(String),
    defect_locations Array(String),
    defect_volumes Array(Float64),
    
    -- Dimensional measurements
    dimensional_accuracy Nullable(Float64),
    measurement_uncertainty Nullable(Float64),
    measurement_method Nullable(String),
    measurement_equipment Nullable(String),
    
    -- User and session data
    user_id Nullable(String),
    session_id Nullable(String),
    operator_id Nullable(String),
    
    -- Metadata
    created_at DateTime,
    
    -- Indexes for performance
    INDEX idx_scan_id scan_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_process_id process_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_scan_type scan_type TYPE bloom_filter GRANULARITY 1,
    INDEX idx_defect_severity defect_severity TYPE bloom_filter GRANULARITY 1,
    INDEX idx_timestamp timestamp TYPE minmax GRANULARITY 1
) 
ENGINE = MergeTree()
ORDER BY (timestamp, scan_id)
PARTITION BY toYYYYMM(timestamp)
TTL timestamp + INTERVAL 2 YEAR
SETTINGS index_granularity = 8192;

-- Materialized view for CT scan analytics
CREATE MATERIALIZED VIEW IF NOT EXISTS ct_scan_analytics
ENGINE = SummingMergeTree()
ORDER BY (scan_type, date)
AS SELECT
    scan_type,
    toStartOfDay(timestamp) as date,
    COUNT(*) as scan_count,
    AVG(file_size) as avg_file_size,
    AVG(image_quality_score) as avg_quality,
    AVG(noise_level) as avg_noise,
    AVG(contrast_ratio) as avg_contrast,
    SUM(defect_count) as total_defects,
    COUNTIf(defect_severity = 'critical') as critical_defects,
    uniq(process_id) as unique_processes
FROM ct_scan_images
GROUP BY scan_type, toStartOfDay(timestamp);
