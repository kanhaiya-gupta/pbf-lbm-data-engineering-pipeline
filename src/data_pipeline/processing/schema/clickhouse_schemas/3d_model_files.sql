-- ClickHouse Schema for 3D Model Files Data (from MongoDB)
-- Optimized for file metadata and 3D model analytics

CREATE TABLE IF NOT EXISTS 3d_model_files (
    -- Primary identifiers
    id UInt64,
    model_id String,
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
    
    -- 3D model metadata
    model_type Nullable(String),
    model_version Nullable(String),
    model_units Nullable(String),
    model_scale Nullable(Float64),
    model_rotation_x Nullable(Float64),
    model_rotation_y Nullable(Float64),
    model_rotation_z Nullable(Float64),
    model_translation_x Nullable(Float64),
    model_translation_y Nullable(Float64),
    model_translation_z Nullable(Float64),
    
    -- Geometry information
    vertex_count Nullable(UInt32),
    face_count Nullable(UInt32),
    edge_count Nullable(UInt32),
    bounding_box_min_x Nullable(Float64),
    bounding_box_min_y Nullable(Float64),
    bounding_box_min_z Nullable(Float64),
    bounding_box_max_x Nullable(Float64),
    bounding_box_max_y Nullable(Float64),
    bounding_box_max_z Nullable(Float64),
    volume Nullable(Float64),
    surface_area Nullable(Float64),
    
    -- Quality metrics
    model_quality_score Nullable(Float64),
    mesh_quality Nullable(String),
    watertight Nullable(UInt8),
    manifold Nullable(UInt8),
    self_intersecting Nullable(UInt8),
    
    -- Processing information
    processing_status Nullable(String),
    processing_algorithm Nullable(String),
    processing_parameters Nullable(String),
    processing_duration Nullable(Float64),
    processing_timestamp Nullable(DateTime),
    
    -- User and session data
    user_id Nullable(String),
    session_id Nullable(String),
    upload_source Nullable(String),
    
    -- Metadata
    created_at DateTime,
    
    -- Indexes for performance
    INDEX idx_model_id model_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_process_id process_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_model_type model_type TYPE bloom_filter GRANULARITY 1,
    INDEX idx_file_status file_status TYPE bloom_filter GRANULARITY 1,
    INDEX idx_timestamp timestamp TYPE minmax GRANULARITY 1
) 
ENGINE = MergeTree()
ORDER BY (timestamp, model_id)
PARTITION BY toYYYYMM(timestamp)
TTL timestamp + INTERVAL 2 YEAR
SETTINGS index_granularity = 8192;

-- Materialized view for model analytics
CREATE MATERIALIZED VIEW IF NOT EXISTS model_analytics
ENGINE = SummingMergeTree()
ORDER BY (model_type, date)
AS SELECT
    model_type,
    toStartOfDay(timestamp) as date,
    COUNT(*) as model_count,
    AVG(file_size) as avg_file_size,
    AVG(vertex_count) as avg_vertex_count,
    AVG(face_count) as avg_face_count,
    AVG(model_quality_score) as avg_quality,
    COUNTIf(watertight = 1) as watertight_count,
    COUNTIf(manifold = 1) as manifold_count,
    uniq(process_id) as unique_processes
FROM model_files
GROUP BY model_type, toStartOfDay(timestamp);
