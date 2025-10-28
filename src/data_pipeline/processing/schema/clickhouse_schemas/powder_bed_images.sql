-- ClickHouse Schema for Powder Bed Images Data (from MongoDB)
-- Optimized for image metadata and powder bed analytics

CREATE TABLE IF NOT EXISTS powder_bed_images (
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
    
    -- Powder bed information
    powder_bed_temperature Nullable(Float64),
    powder_bed_pressure Nullable(Float64),
    powder_bed_humidity Nullable(Float64),
    powder_bed_level Nullable(Float64),
    powder_bed_quality Nullable(String),
    
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
    
    -- Powder analysis
    powder_coverage Nullable(Float64),
    powder_distribution Nullable(String),
    powder_particle_size Nullable(Float64),
    powder_flowability Nullable(Float64),
    
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
    INDEX idx_defect_severity defect_severity TYPE bloom_filter GRANULARITY 1,
    INDEX idx_timestamp timestamp TYPE minmax GRANULARITY 1
) 
ENGINE = MergeTree()
ORDER BY (timestamp, image_id)
PARTITION BY toYYYYMM(timestamp)
TTL timestamp + INTERVAL 2 YEAR
SETTINGS index_granularity = 8192;

-- Materialized view for powder bed analytics
CREATE MATERIALIZED VIEW IF NOT EXISTS powder_bed_analytics
ENGINE = SummingMergeTree()
ORDER BY (image_type, date)
AS SELECT
    image_type,
    toStartOfDay(timestamp) as date,
    COUNT(*) as image_count,
    AVG(file_size) as avg_file_size,
    AVG(image_quality_score) as avg_quality,
    AVG(powder_coverage) as avg_coverage,
    AVG(powder_particle_size) as avg_particle_size,
    SUM(defect_count) as total_defects,
    COUNTIf(defect_severity = 'critical') as critical_defects,
    uniq(process_id) as unique_processes
FROM powder_bed_images
GROUP BY image_type, toStartOfDay(timestamp);
