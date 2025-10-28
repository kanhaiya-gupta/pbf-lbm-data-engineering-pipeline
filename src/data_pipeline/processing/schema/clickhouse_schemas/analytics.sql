-- ClickHouse Schema for Analytics Data
-- Optimized for business intelligence and predictive analytics

CREATE TABLE IF NOT EXISTS analytics (
    -- Primary identifiers
    id UInt64,
    analytics_id String,
    timestamp DateTime,
    
    -- Analysis metadata
    analysis_type Nullable(String),
    data_source Nullable(String),
    
    -- Performance metrics
    throughput Nullable(Float64),
    efficiency Nullable(Float64),
    utilization Nullable(Float64),
    downtime Nullable(UInt32),
    uptime Nullable(UInt32),
    
    -- Quality analytics
    defect_rate Nullable(Float64),
    quality_score Nullable(Float64),
    rework_rate Nullable(Float64),
    scrap_rate Nullable(Float64),
    first_pass_yield Nullable(Float64),
    
    -- Cost analytics
    material_cost Nullable(Float64),
    energy_cost Nullable(Float64),
    labor_cost Nullable(Float64),
    maintenance_cost Nullable(Float64),
    total_cost Nullable(Float64),
    
    -- Trend analysis
    trend_direction Nullable(String),
    trend_magnitude Nullable(Float64),
    trend_confidence Nullable(Float64),
    has_seasonality Nullable(UInt8),
    seasonal_period Nullable(UInt32),
    seasonal_strength Nullable(Float64),
    
    -- Predictive analytics
    prediction_type Nullable(String),
    predicted_value Nullable(Float64),
    lower_bound Nullable(Float64),
    upper_bound Nullable(Float64),
    prediction_horizon Nullable(UInt32),
    model_accuracy Nullable(Float64),
    
    -- Anomaly detection
    anomaly_score Nullable(Float64),
    anomaly_type Nullable(String),
    severity Nullable(String),
    detection_method Nullable(String),
    affected_metrics Array(String),
    
    -- Comparative analysis
    baseline_start_date Nullable(DateTime),
    baseline_end_date Nullable(DateTime),
    comparison_start_date Nullable(DateTime),
    comparison_end_date Nullable(DateTime),
    throughput_change Nullable(Float64),
    quality_change Nullable(Float64),
    cost_change Nullable(Float64),
    
    -- KPI metrics
    kpi_name Nullable(String),
    kpi_value Nullable(Float64),
    kpi_target Nullable(Float64),
    kpi_status Nullable(String),
    
    -- Metadata
    created_at DateTime,
    
    -- Indexes for performance
    INDEX idx_analytics_id analytics_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_analysis_type analysis_type TYPE bloom_filter GRANULARITY 1,
    INDEX idx_anomaly_type anomaly_type TYPE bloom_filter GRANULARITY 1,
    INDEX idx_timestamp timestamp TYPE minmax GRANULARITY 1
) 
ENGINE = MergeTree()
ORDER BY (timestamp, analytics_id)
PARTITION BY toYYYYMM(timestamp)
TTL timestamp + INTERVAL 1 YEAR
SETTINGS index_granularity = 8192;

-- Materialized view for cost analysis
CREATE MATERIALIZED VIEW IF NOT EXISTS cost_analytics_summary
ENGINE = SummingMergeTree()
ORDER BY (analysis_type, date)
AS SELECT
    analysis_type,
    toStartOfDay(timestamp) as date,
    COUNT(*) as analysis_count,
    AVG(total_cost) as avg_total_cost,
    AVG(material_cost) as avg_material_cost,
    AVG(energy_cost) as avg_energy_cost,
    AVG(labor_cost) as avg_labor_cost,
    AVG(maintenance_cost) as avg_maintenance_cost,
    SUM(total_cost) as total_cost_sum
FROM analytics
GROUP BY analysis_type, toStartOfDay(timestamp);

-- Materialized view for anomaly detection trends
CREATE MATERIALIZED VIEW IF NOT EXISTS anomaly_detection_trends
ENGINE = SummingMergeTree()
ORDER BY (anomaly_type, severity, date)
AS SELECT
    anomaly_type,
    severity,
    toStartOfDay(timestamp) as date,
    COUNT(*) as anomaly_count,
    AVG(anomaly_score) as avg_anomaly_score,
    AVG(quality_score) as avg_quality_during_anomaly,
    AVG(throughput) as avg_throughput_during_anomaly
FROM analytics
WHERE anomaly_score > 0
GROUP BY anomaly_type, severity, toStartOfDay(timestamp);
