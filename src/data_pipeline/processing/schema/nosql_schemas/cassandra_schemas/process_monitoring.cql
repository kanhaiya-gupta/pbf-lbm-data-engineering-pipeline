-- Process Monitoring Events Table
-- Optimized for process event tracking and monitoring

CREATE TABLE IF NOT EXISTS process_monitoring (
    -- Partition key: process_id for efficient process-based queries
    process_id text,
    
    -- Clustering key: timestamp for chronological ordering
    timestamp timestamp,
    
    -- Machine and operator context
    machine_id text,
    operator_id text,
    
    -- Event data
    event_type text,
    event_data text, -- JSON string for Dict[str, Any]
    severity text, -- AlertSeverity enum as text
    
    -- Process context
    process_status text, -- ProcessStatus enum as text
    layer_number int,
    progress_percentage double,
    
    -- Session and metadata
    session_id text,
    metadata text, -- JSON string for Dict[str, Any]
    
    -- Record metadata
    created_at timestamp,
    
    -- Primary key definition
    PRIMARY KEY (process_id, timestamp)
) WITH CLUSTERING ORDER BY (timestamp DESC);

-- Create indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_process_monitoring_machine_id ON process_monitoring (machine_id);
CREATE INDEX IF NOT EXISTS idx_process_monitoring_event_type ON process_monitoring (event_type);
CREATE INDEX IF NOT EXISTS idx_process_monitoring_severity ON process_monitoring (severity);
CREATE INDEX IF NOT EXISTS idx_process_monitoring_process_status ON process_monitoring (process_status);
CREATE INDEX IF NOT EXISTS idx_process_monitoring_operator_id ON process_monitoring (operator_id);

-- Create materialized view for machine-based queries
CREATE MATERIALIZED VIEW IF NOT EXISTS process_monitoring_by_machine AS
    SELECT machine_id, timestamp, process_id, event_type, severity, process_status, operator_id
    FROM process_monitoring
    WHERE machine_id IS NOT NULL AND timestamp IS NOT NULL AND process_id IS NOT NULL
    PRIMARY KEY (machine_id, timestamp, process_id)
    WITH CLUSTERING ORDER BY (timestamp DESC);

-- Create materialized view for event type queries
CREATE MATERIALIZED VIEW IF NOT EXISTS process_monitoring_by_event_type AS
    SELECT event_type, timestamp, process_id, machine_id, severity, process_status
    FROM process_monitoring
    WHERE event_type IS NOT NULL AND timestamp IS NOT NULL AND process_id IS NOT NULL
    PRIMARY KEY (event_type, timestamp, process_id)
    WITH CLUSTERING ORDER BY (timestamp DESC);

-- Create materialized view for severity-based queries
CREATE MATERIALIZED VIEW IF NOT EXISTS process_monitoring_by_severity AS
    SELECT severity, timestamp, process_id, machine_id, event_type, process_status
    FROM process_monitoring
    WHERE severity IS NOT NULL AND timestamp IS NOT NULL AND process_id IS NOT NULL
    PRIMARY KEY (severity, timestamp, process_id)
    WITH CLUSTERING ORDER BY (timestamp DESC);

-- Set table properties for time-series optimization
ALTER TABLE process_monitoring WITH
    compaction = {'class': 'TimeWindowCompactionStrategy', 'compaction_window_unit': 'DAYS', 'compaction_window_size': 1}
    AND compression = {'sstable_compression': 'LZ4Compressor'}
    AND gc_grace_seconds = 864000; -- 10 days
