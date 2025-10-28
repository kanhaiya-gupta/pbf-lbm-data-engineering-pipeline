-- ClickHouse Schema for Build Instructions Data (from MongoDB)
-- Optimized for process documentation and instruction analytics

CREATE TABLE IF NOT EXISTS build_instructions (
    -- Primary identifiers
    id UInt64,
    instruction_id String,
    build_id Nullable(String),
    process_id Nullable(String),
    part_id Nullable(String),
    machine_id Nullable(String),
    timestamp DateTime,
    
    -- Instruction metadata
    instruction_type Nullable(String),
    instruction_content Nullable(String),
    instruction_version Nullable(String),
    instruction_status Nullable(String),
    instruction_priority Nullable(UInt32),
    
    -- Layer information
    layer_number Nullable(UInt32),
    layer_thickness Nullable(Float64),
    laser_power Nullable(Float64),
    scan_speed Nullable(Float64),
    hatch_spacing Nullable(Float64),
    exposure_time Nullable(Float64),
    
    -- Geometry data
    contour_paths Nullable(String),
    hatch_patterns Nullable(String),
    support_structures Nullable(String),
    support_type Nullable(String),
    support_density Nullable(Float64),
    
    -- Material requirements
    material_type Nullable(String),
    powder_type Nullable(String),
    powder_amount Nullable(Float64),
    powder_condition Nullable(String),
    plate_material Nullable(String),
    plate_temperature Nullable(Float64),
    plate_preparation Nullable(String),
    
    -- Quality requirements
    dimensional_tolerance Nullable(Float64),
    surface_roughness Nullable(Float64),
    density_requirement Nullable(Float64),
    tensile_strength Nullable(Float64),
    yield_strength Nullable(Float64),
    hardness Nullable(Float64),
    
    -- Process parameters
    build_temperature Nullable(Float64),
    chamber_atmosphere Nullable(String),
    oxygen_level Nullable(Float64),
    build_speed Nullable(Float64),
    cooling_rate Nullable(Float64),
    
    -- User and session data
    user_id Nullable(String),
    session_id Nullable(String),
    created_by Nullable(String),
    
    -- Metadata
    created_at DateTime,
    
    -- Indexes for performance
    INDEX idx_instruction_id instruction_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_build_id build_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_instruction_type instruction_type TYPE bloom_filter GRANULARITY 1,
    INDEX idx_material_type material_type TYPE bloom_filter GRANULARITY 1,
    INDEX idx_timestamp timestamp TYPE minmax GRANULARITY 1
) 
ENGINE = MergeTree()
ORDER BY (timestamp, instruction_id)
PARTITION BY toYYYYMM(timestamp)
TTL timestamp + INTERVAL 2 YEAR
SETTINGS index_granularity = 8192;

-- Materialized view for instruction analytics
CREATE MATERIALIZED VIEW IF NOT EXISTS instruction_analytics
ENGINE = SummingMergeTree()
ORDER BY (instruction_type, material_type, date)
AS SELECT
    instruction_type,
    material_type,
    toStartOfDay(timestamp) as date,
    COUNT(*) as instruction_count,
    AVG(layer_thickness) as avg_layer_thickness,
    AVG(laser_power) as avg_laser_power,
    AVG(scan_speed) as avg_scan_speed,
    AVG(dimensional_tolerance) as avg_tolerance,
    AVG(surface_roughness) as avg_roughness,
    uniq(build_id) as unique_builds,
    uniq(machine_id) as unique_machines
FROM build_instructions
GROUP BY instruction_type, material_type, toStartOfDay(timestamp);
