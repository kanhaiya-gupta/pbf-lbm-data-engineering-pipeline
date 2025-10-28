-- ClickHouse Schema for Machine Status Data
-- Optimized for real-time monitoring and predictive maintenance

CREATE TABLE IF NOT EXISTS machine_status (
    -- Primary identifiers
    id UInt64,
    machine_id String,
    timestamp DateTime,
    
    -- Operational state
    status Nullable(String),
    current_state Nullable(String),
    previous_state Nullable(String),
    state_duration Nullable(UInt32),
    state_transitions Nullable(UInt32),
    
    -- System health metrics
    overall_health Nullable(Float64),
    cpu_usage Nullable(Float64),
    memory_usage Nullable(Float64),
    disk_usage Nullable(Float64),
    network_status Nullable(String),
    
    -- Laser system metrics
    laser_power Nullable(Float64),
    laser_temperature Nullable(Float64),
    laser_status Nullable(String),
    laser_hours Nullable(UInt32),
    laser_wavelength Nullable(Float64),
    
    -- Build platform metrics
    platform_temperature Nullable(Float64),
    platform_x Nullable(Float64),
    platform_y Nullable(Float64),
    platform_z Nullable(Float64),
    platform_status Nullable(String),
    
    -- Powder system metrics
    powder_level Nullable(Float64),
    powder_temperature Nullable(Float64),
    powder_flow_rate Nullable(Float64),
    powder_status Nullable(String),
    powder_quality Nullable(String),
    
    -- Environmental conditions
    chamber_temperature Nullable(Float64),
    chamber_humidity Nullable(Float64),
    oxygen_level Nullable(Float64),
    pressure Nullable(Float64),
    
    -- Alerts and warnings
    active_alerts Nullable(UInt32),
    alert_level Nullable(String),
    alert_types Array(String),
    last_maintenance Nullable(DateTime),
    next_maintenance Nullable(DateTime),
    
    -- Performance metrics
    throughput Nullable(Float64),
    efficiency Nullable(Float64),
    utilization Nullable(Float64),
    downtime Nullable(UInt32),
    uptime Nullable(UInt32),
    
    -- Maintenance information
    maintenance_due Nullable(UInt8),
    maintenance_type Nullable(String),
    maintenance_interval Nullable(UInt32),
    last_service_date Nullable(DateTime),
    service_history Array(String),
    
    -- Metadata
    created_at DateTime,
    
    -- Indexes for performance
    INDEX idx_machine_id machine_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_status status TYPE bloom_filter GRANULARITY 1,
    INDEX idx_alert_level alert_level TYPE bloom_filter GRANULARITY 1,
    INDEX idx_timestamp timestamp TYPE minmax GRANULARITY 1
) 
ENGINE = MergeTree()
ORDER BY (timestamp, machine_id)
PARTITION BY toYYYYMM(timestamp)
TTL timestamp + INTERVAL 1 YEAR
SETTINGS index_granularity = 8192;

-- Materialized view for machine health trends
CREATE MATERIALIZED VIEW IF NOT EXISTS machine_health_trends
ENGINE = SummingMergeTree()
ORDER BY (machine_id, hour)
AS SELECT
    machine_id,
    toStartOfHour(timestamp) as hour,
    AVG(overall_health) as avg_health,
    AVG(cpu_usage) as avg_cpu,
    AVG(memory_usage) as avg_memory,
    AVG(disk_usage) as avg_disk,
    AVG(throughput) as avg_throughput,
    AVG(efficiency) as avg_efficiency,
    SUM(downtime) as total_downtime,
    COUNT(*) as status_records
FROM machine_status
GROUP BY machine_id, toStartOfHour(timestamp);

-- Materialized view for alert analysis
CREATE MATERIALIZED VIEW IF NOT EXISTS machine_alert_analysis
ENGINE = SummingMergeTree()
ORDER BY (machine_id, alert_level, date)
AS SELECT
    machine_id,
    alert_level,
    toStartOfDay(timestamp) as date,
    COUNT(*) as alert_count,
    AVG(overall_health) as avg_health_during_alerts,
    COUNTIf(maintenance_due = 1) as maintenance_due_count
FROM machine_status
WHERE active_alerts > 0
GROUP BY machine_id, alert_level, toStartOfDay(timestamp);
