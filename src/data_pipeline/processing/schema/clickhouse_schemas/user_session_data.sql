-- ClickHouse Schema for User Session Data (from Redis)
-- Optimized for user analytics and session management

CREATE TABLE IF NOT EXISTS user_session_data (
    -- Primary identifiers
    id UInt64,
    session_id String,
    cache_key Nullable(String),
    timestamp DateTime,
    
    -- User information
    user_id Nullable(String),
    username Nullable(String),
    user_role Nullable(String),
    user_permissions Array(String),
    user_department Nullable(String),
    user_team Nullable(String),
    
    -- Session information
    session_status Nullable(String),
    session_type Nullable(String),
    session_duration Nullable(Float64),
    session_start_time Nullable(DateTime),
    session_end_time Nullable(DateTime),
    last_activity Nullable(DateTime),
    
    -- Authentication data
    login_method Nullable(String),
    login_ip Nullable(String),
    login_location Nullable(String),
    login_device Nullable(String),
    login_browser Nullable(String),
    login_os Nullable(String),
    
    -- Session analytics
    page_views Nullable(UInt32),
    api_calls Nullable(UInt32),
    data_queries Nullable(UInt32),
    file_downloads Nullable(UInt32),
    session_actions Nullable(UInt32),
    
    -- Performance metrics
    response_time Nullable(Float64),
    memory_usage Nullable(UInt64),
    cpu_usage Nullable(Float64),
    network_usage Nullable(UInt64),
    
    -- Security data
    security_level Nullable(String),
    risk_score Nullable(Float64),
    suspicious_activity Nullable(UInt8),
    failed_attempts Nullable(UInt32),
    last_password_change Nullable(DateTime),
    
    -- Session context
    active_processes Array(String),
    active_machines Array(String),
    active_builds Array(String),
    current_workspace Nullable(String),
    
    -- Metadata
    created_at DateTime,
    
    -- Indexes for performance
    INDEX idx_session_id session_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_user_id user_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_session_status session_status TYPE bloom_filter GRANULARITY 1,
    INDEX idx_user_role user_role TYPE bloom_filter GRANULARITY 1,
    INDEX idx_timestamp timestamp TYPE minmax GRANULARITY 1
) 
ENGINE = MergeTree()
ORDER BY (timestamp, session_id)
PARTITION BY toYYYYMM(timestamp)
TTL timestamp + INTERVAL 6 MONTH
SETTINGS index_granularity = 8192;

-- Materialized view for user session analytics
CREATE MATERIALIZED VIEW IF NOT EXISTS user_session_analytics
ENGINE = SummingMergeTree()
ORDER BY (user_role, session_status, date)
AS SELECT
    user_role,
    session_status,
    toStartOfDay(timestamp) as date,
    COUNT(*) as session_count,
    AVG(session_duration) as avg_session_duration,
    AVG(page_views) as avg_page_views,
    AVG(api_calls) as avg_api_calls,
    AVG(response_time) as avg_response_time,
    COUNTIf(suspicious_activity = 1) as suspicious_sessions,
    uniq(user_id) as unique_users,
    uniq(session_id) as unique_sessions
FROM user_session_data
GROUP BY user_role, session_status, toStartOfDay(timestamp);
