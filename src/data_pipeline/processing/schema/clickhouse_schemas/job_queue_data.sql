-- ClickHouse Schema for Job Queue Data (from Redis)
-- Optimized for job processing analytics and queue management

CREATE TABLE IF NOT EXISTS job_queue_data (
    -- Primary identifiers
    id UInt64,
    job_id String,
    cache_key Nullable(String),
    timestamp DateTime,
    
    -- Job information
    job_type Nullable(String),
    job_status Nullable(String),
    job_priority Nullable(UInt32),
    job_category Nullable(String),
    job_description Nullable(String),
    
    -- Process and machine context
    process_id Nullable(String),
    machine_id Nullable(String),
    build_id Nullable(String),
    part_id Nullable(String),
    
    -- Job execution data
    execution_status Nullable(String),
    execution_start_time Nullable(DateTime),
    execution_end_time Nullable(DateTime),
    execution_duration Nullable(Float64),
    execution_attempts Nullable(UInt32),
    max_attempts Nullable(UInt32),
    
    -- Performance metrics
    cpu_usage Nullable(Float64),
    memory_usage Nullable(UInt64),
    disk_usage Nullable(UInt64),
    network_usage Nullable(UInt64),
    processing_time Nullable(Float64),
    queue_wait_time Nullable(Float64),
    
    -- Job parameters
    job_parameters Nullable(String),
    job_input_data Nullable(String),
    job_output_data Nullable(String),
    job_dependencies Array(String),
    job_requirements Nullable(String),
    
    -- User and session data
    user_id Nullable(String),
    session_id Nullable(String),
    created_by Nullable(String),
    assigned_to Nullable(String),
    
    -- Queue analytics
    queue_position Nullable(UInt32),
    queue_priority Nullable(UInt32),
    estimated_completion_time Nullable(DateTime),
    actual_completion_time Nullable(DateTime),
    
    -- Error handling
    error_count Nullable(UInt32),
    error_messages Array(String),
    last_error Nullable(String),
    retry_count Nullable(UInt32),
    
    -- Metadata
    created_at DateTime,
    
    -- Indexes for performance
    INDEX idx_job_id job_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_job_type job_type TYPE bloom_filter GRANULARITY 1,
    INDEX idx_job_status job_status TYPE bloom_filter GRANULARITY 1,
    INDEX idx_process_id process_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_timestamp timestamp TYPE minmax GRANULARITY 1
) 
ENGINE = MergeTree()
ORDER BY (timestamp, job_id)
PARTITION BY toYYYYMM(timestamp)
TTL timestamp + INTERVAL 3 MONTH
SETTINGS index_granularity = 8192;

-- Materialized view for job queue analytics
CREATE MATERIALIZED VIEW IF NOT EXISTS job_queue_analytics
ENGINE = SummingMergeTree()
ORDER BY (job_type, job_status, hour)
AS SELECT
    job_type,
    job_status,
    toStartOfHour(timestamp) as hour,
    COUNT(*) as job_count,
    AVG(execution_duration) as avg_execution_time,
    AVG(queue_wait_time) as avg_queue_wait,
    AVG(cpu_usage) as avg_cpu_usage,
    AVG(memory_usage) as avg_memory_usage,
    SUM(error_count) as total_errors,
    COUNTIf(execution_status = 'completed') as completed_jobs,
    COUNTIf(execution_status = 'failed') as failed_jobs,
    uniq(process_id) as unique_processes,
    uniq(machine_id) as unique_machines
FROM job_queue_data
GROUP BY job_type, job_status, toStartOfHour(timestamp);
