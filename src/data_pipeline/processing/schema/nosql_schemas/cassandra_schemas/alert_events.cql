-- Alert Events Table
-- Optimized for system alerts and notifications

CREATE TABLE IF NOT EXISTS alert_events (
    -- Partition key: alert_id for efficient alert-based queries
    alert_id text,
    
    -- Clustering key: timestamp for chronological ordering
    timestamp timestamp,
    
    -- Context identifiers
    process_id text,
    machine_id text,
    sensor_id text,
    
    -- Alert data
    alert_type text,
    severity text,
    title text,
    description text,
    
    -- Alert context
    threshold_value double,
    actual_value double,
    tolerance double,
    
    -- Alert lifecycle
    status text,
    acknowledged boolean,
    acknowledged_by text,
    acknowledged_at timestamp,
    resolved boolean,
    resolved_at timestamp,
    
    -- System metadata
    source_system text,
    metadata text, -- JSON string for Dict[str, Any]
    
    -- Record metadata
    created_at timestamp,
    
    -- Primary key definition
    PRIMARY KEY (alert_id, timestamp)
) WITH CLUSTERING ORDER BY (timestamp DESC);

-- Create indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_alert_events_process_id ON alert_events (process_id);
CREATE INDEX IF NOT EXISTS idx_alert_events_machine_id ON alert_events (machine_id);
CREATE INDEX IF NOT EXISTS idx_alert_events_sensor_id ON alert_events (sensor_id);
CREATE INDEX IF NOT EXISTS idx_alert_events_alert_type ON alert_events (alert_type);
CREATE INDEX IF NOT EXISTS idx_alert_events_severity ON alert_events (severity);
CREATE INDEX IF NOT EXISTS idx_alert_events_status ON alert_events (status);
CREATE INDEX IF NOT EXISTS idx_alert_events_acknowledged ON alert_events (acknowledged);
CREATE INDEX IF NOT EXISTS idx_alert_events_resolved ON alert_events (resolved);
CREATE INDEX IF NOT EXISTS idx_alert_events_source_system ON alert_events (source_system);

-- Create materialized view for process-based queries
CREATE MATERIALIZED VIEW IF NOT EXISTS alert_events_by_process AS
    SELECT process_id, timestamp, alert_id, alert_type, severity, status, acknowledged, resolved
    FROM alert_events
    WHERE process_id IS NOT NULL AND timestamp IS NOT NULL AND alert_id IS NOT NULL
    PRIMARY KEY (process_id, timestamp, alert_id)
    WITH CLUSTERING ORDER BY (timestamp DESC);

-- Create materialized view for machine-based queries
CREATE MATERIALIZED VIEW IF NOT EXISTS alert_events_by_machine AS
    SELECT machine_id, timestamp, alert_id, alert_type, severity, status, acknowledged, resolved
    FROM alert_events
    WHERE machine_id IS NOT NULL AND timestamp IS NOT NULL AND alert_id IS NOT NULL
    PRIMARY KEY (machine_id, timestamp, alert_id)
    WITH CLUSTERING ORDER BY (timestamp DESC);

-- Create materialized view for sensor-based queries
CREATE MATERIALIZED VIEW IF NOT EXISTS alert_events_by_sensor AS
    SELECT sensor_id, timestamp, alert_id, alert_type, severity, status, acknowledged, resolved
    FROM alert_events
    WHERE sensor_id IS NOT NULL AND timestamp IS NOT NULL AND alert_id IS NOT NULL
    PRIMARY KEY (sensor_id, timestamp, alert_id)
    WITH CLUSTERING ORDER BY (timestamp DESC);

-- Create materialized view for alert type queries
CREATE MATERIALIZED VIEW IF NOT EXISTS alert_events_by_type AS
    SELECT alert_type, timestamp, alert_id, process_id, machine_id, severity, status
    FROM alert_events
    WHERE alert_type IS NOT NULL AND timestamp IS NOT NULL AND alert_id IS NOT NULL
    PRIMARY KEY (alert_type, timestamp, alert_id)
    WITH CLUSTERING ORDER BY (timestamp DESC);

-- Create materialized view for severity-based queries
CREATE MATERIALIZED VIEW IF NOT EXISTS alert_events_by_severity AS
    SELECT severity, timestamp, alert_id, process_id, machine_id, alert_type, status
    FROM alert_events
    WHERE severity IS NOT NULL AND timestamp IS NOT NULL AND alert_id IS NOT NULL
    PRIMARY KEY (severity, timestamp, alert_id)
    WITH CLUSTERING ORDER BY (timestamp DESC);

-- Create materialized view for status-based queries
CREATE MATERIALIZED VIEW IF NOT EXISTS alert_events_by_status AS
    SELECT status, timestamp, alert_id, process_id, machine_id, alert_type, severity
    FROM alert_events
    WHERE status IS NOT NULL AND timestamp IS NOT NULL AND alert_id IS NOT NULL
    PRIMARY KEY (status, timestamp, alert_id)
    WITH CLUSTERING ORDER BY (timestamp DESC);

-- Create materialized view for unacknowledged alerts
CREATE MATERIALIZED VIEW IF NOT EXISTS alert_events_unacknowledged AS
    SELECT acknowledged, timestamp, alert_id, process_id, machine_id, alert_type, severity
    FROM alert_events
    WHERE acknowledged IS NOT NULL AND timestamp IS NOT NULL AND alert_id IS NOT NULL
    PRIMARY KEY (acknowledged, timestamp, alert_id)
    WITH CLUSTERING ORDER BY (timestamp DESC);

-- Set table properties for time-series optimization
ALTER TABLE alert_events WITH
    compaction = {'class': 'TimeWindowCompactionStrategy', 'compaction_window_unit': 'DAYS', 'compaction_window_size': 1}
    AND compression = {'sstable_compression': 'LZ4Compressor'}
    AND gc_grace_seconds = 864000; -- 10 days
