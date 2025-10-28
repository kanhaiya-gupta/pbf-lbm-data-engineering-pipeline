-- ClickHouse Schema for Process Logs Data (from MongoDB)
-- Optimized for log analytics and operational monitoring

CREATE TABLE IF NOT EXISTS process_logs (
    -- Primary identifiers
    id UInt64,
    log_id String,
    process_id Nullable(String),
    build_id Nullable(String),
    part_id Nullable(String),
    machine_id Nullable(String),
    timestamp DateTime,
    
    -- Log information
    log_level Nullable(String),
    log_message Nullable(String),
    source_module Nullable(String),
    event_type Nullable(String),
    user_id Nullable(String),
    
    -- Annotation data
    annotation_type Nullable(String),
    annotation_text Nullable(String),
    related_documents Array(String),
    
    -- Session metadata
    session_id Nullable(String),
    ip_address Nullable(String),
    user_agent Nullable(String),
    
    -- Data quality
    data_quality Nullable(String),
    validation_status Nullable(String),
    generated_by Nullable(String),
    generation_timestamp Nullable(DateTime),
    
    -- Tags and relationships
    tags Array(String),
    related_processes Array(String),
    related_builds Array(String),
    related_parts Array(String),
    
    -- Metadata
    created_at DateTime,
    
    -- Indexes for performance
    INDEX idx_log_id log_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_process_id process_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_log_level log_level TYPE bloom_filter GRANULARITY 1,
    INDEX idx_event_type event_type TYPE bloom_filter GRANULARITY 1,
    INDEX idx_timestamp timestamp TYPE minmax GRANULARITY 1
) 
ENGINE = MergeTree()
ORDER BY (timestamp, log_id)
PARTITION BY toYYYYMM(timestamp)
TTL timestamp + INTERVAL 1 YEAR
SETTINGS index_granularity = 8192;

-- Materialized view for log analytics
CREATE MATERIALIZED VIEW IF NOT EXISTS log_analytics
ENGINE = SummingMergeTree()
ORDER BY (log_level, event_type, hour)
AS SELECT
    log_level,
    event_type,
    toStartOfHour(timestamp) as hour,
    COUNT(*) as log_count,
    COUNTIf(validation_status = 'passed') as valid_logs,
    COUNTIf(validation_status = 'failed') as invalid_logs,
    uniq(process_id) as unique_processes,
    uniq(machine_id) as unique_machines
FROM process_logs
GROUP BY log_level, event_type, toStartOfHour(timestamp);

-- Materialized view for error analysis
CREATE MATERIALIZED VIEW IF NOT EXISTS error_analysis
ENGINE = SummingMergeTree()
ORDER BY (source_module, date)
AS SELECT
    source_module,
    toStartOfDay(timestamp) as date,
    COUNT(*) as total_logs,
    COUNTIf(log_level = 'ERROR') as error_count,
    COUNTIf(log_level = 'WARNING') as warning_count,
    COUNTIf(validation_status = 'failed') as validation_failures,
    uniq(process_id) as affected_processes
FROM process_logs
GROUP BY source_module, toStartOfDay(timestamp);
