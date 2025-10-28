-- ClickHouse Schema for PBF Process Data
-- Optimized for analytics and time-series queries

CREATE TABLE IF NOT EXISTS pbf_processes (
    -- Primary identifiers
    id UInt64,
    process_id String,
    build_id Nullable(String),
    part_id Nullable(String),
    
    -- Timestamps
    timestamp DateTime,
    created_at DateTime,
    updated_at Nullable(DateTime),
    
    -- Process parameters
    laser_power Nullable(Float64),
    scan_speed Nullable(Float64),
    layer_thickness Nullable(Float64),
    hatch_spacing Nullable(Float64),
    build_plate_temp Nullable(Float64),
    exposure_time Nullable(Float64),
    focus_offset Nullable(Float64),
    
    -- Material information
    material_type Nullable(String),
    powder_batch_id Nullable(String),
    powder_condition Nullable(String),
    powder_particle_size Nullable(Float64),
    powder_flowability Nullable(Float64),
    
    -- Quality metrics
    density Nullable(Float64),
    surface_roughness Nullable(Float64),
    dimensional_accuracy Nullable(Float64),
    defect_count Nullable(UInt32),
    quality_score Nullable(Float64),
    quality_status Nullable(String),
    
    -- Operational metadata
    operator_id Nullable(String),
    machine_id Nullable(String),
    build_job_id Nullable(String),
    
    -- Indexes for performance
    INDEX idx_process_id process_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_machine_id machine_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_material_type material_type TYPE bloom_filter GRANULARITY 1,
    INDEX idx_quality_status quality_status TYPE bloom_filter GRANULARITY 1,
    INDEX idx_timestamp timestamp TYPE minmax GRANULARITY 1
) 
ENGINE = MergeTree()
ORDER BY (timestamp, process_id)
PARTITION BY toYYYYMM(timestamp)
TTL timestamp + INTERVAL 2 YEAR
SETTINGS index_granularity = 8192;

-- Materialized view for quality analytics
CREATE MATERIALIZED VIEW IF NOT EXISTS pbf_quality_analytics
ENGINE = SummingMergeTree()
ORDER BY (material_type, date)
AS SELECT
    material_type,
    toStartOfDay(timestamp) as date,
    COUNT(*) as process_count,
    AVG(quality_score) as avg_quality_score,
    AVG(density) as avg_density,
    AVG(surface_roughness) as avg_surface_roughness,
    SUM(defect_count) as total_defects,
    COUNTIf(quality_status = 'acceptable') as acceptable_count,
    COUNTIf(quality_status = 'rejected') as rejected_count
FROM pbf_processes
GROUP BY material_type, toStartOfDay(timestamp);

-- Materialized view for process performance
CREATE MATERIALIZED VIEW IF NOT EXISTS pbf_process_performance
ENGINE = SummingMergeTree()
ORDER BY (machine_id, hour)
AS SELECT
    machine_id,
    toStartOfHour(timestamp) as hour,
    COUNT(*) as process_count,
    AVG(laser_power) as avg_laser_power,
    AVG(scan_speed) as avg_scan_speed,
    AVG(quality_score) as avg_quality,
    SUM(defect_count) as total_defects
FROM pbf_processes
GROUP BY machine_id, toStartOfHour(timestamp);
