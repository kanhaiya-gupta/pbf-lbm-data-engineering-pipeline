-- ClickHouse Schema for Redis Cache Data
-- Optimized for cache analytics and performance monitoring

CREATE TABLE IF NOT EXISTS redis_cache_data (
    -- Primary identifiers
    id UInt64,
    cache_key String,
    cache_type Nullable(String),
    timestamp DateTime,
    
    -- Cache metadata
    cache_operation Nullable(String),
    cache_hit Nullable(UInt8),
    cache_miss Nullable(UInt8),
    cache_ttl Nullable(UInt32),
    cache_size Nullable(UInt64),
    cache_compression Nullable(String),
    
    -- Performance metrics
    response_time Nullable(Float64),
    memory_usage Nullable(UInt64),
    cpu_usage Nullable(Float64),
    network_latency Nullable(Float64),
    
    -- Data information
    data_type Nullable(String),
    data_size Nullable(UInt64),
    data_format Nullable(String),
    data_compression_ratio Nullable(Float64),
    
    -- User and session data
    user_id Nullable(String),
    session_id Nullable(String),
    client_ip Nullable(String),
    user_agent Nullable(String),
    
    -- Cache analytics
    access_frequency Nullable(UInt32),
    last_accessed Nullable(DateTime),
    expiration_time Nullable(DateTime),
    cache_priority Nullable(UInt32),
    
    -- Metadata
    created_at DateTime,
    
    -- Indexes for performance
    INDEX idx_cache_key cache_key TYPE bloom_filter GRANULARITY 1,
    INDEX idx_cache_type cache_type TYPE bloom_filter GRANULARITY 1,
    INDEX idx_cache_operation cache_operation TYPE bloom_filter GRANULARITY 1,
    INDEX idx_timestamp timestamp TYPE minmax GRANULARITY 1
) 
ENGINE = MergeTree()
ORDER BY (timestamp, cache_key)
PARTITION BY toYYYYMM(timestamp)
TTL timestamp + INTERVAL 1 MONTH
SETTINGS index_granularity = 8192;

-- Materialized view for cache performance analytics
CREATE MATERIALIZED VIEW IF NOT EXISTS cache_performance_analytics
ENGINE = SummingMergeTree()
ORDER BY (cache_type, hour)
AS SELECT
    cache_type,
    toStartOfHour(timestamp) as hour,
    COUNT(*) as total_operations,
    COUNTIf(cache_hit = 1) as cache_hits,
    COUNTIf(cache_miss = 1) as cache_misses,
    AVG(response_time) as avg_response_time,
    AVG(memory_usage) as avg_memory_usage,
    AVG(cpu_usage) as avg_cpu_usage,
    uniq(cache_key) as unique_keys,
    uniq(user_id) as unique_users
FROM redis_cache_data
GROUP BY cache_type, toStartOfHour(timestamp);
