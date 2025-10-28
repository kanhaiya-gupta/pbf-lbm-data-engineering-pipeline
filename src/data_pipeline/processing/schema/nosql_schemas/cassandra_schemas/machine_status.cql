-- Machine Status Updates Table
-- Optimized for machine health monitoring and status tracking

CREATE TABLE IF NOT EXISTS machine_status (
    -- Partition key: machine_id for efficient machine-based queries
    machine_id text,
    
    -- Clustering key: timestamp for chronological ordering
    timestamp timestamp,
    
    -- Operator context
    operator_id text,
    
    -- Status data
    status text,
    previous_status text,
    
    -- Performance metrics
    performance_metrics text, -- JSON string for Dict[str, float]
    health_score double,
    
    -- Alerts and maintenance
    active_alerts list<text>,
    maintenance_required boolean,
    next_maintenance_date timestamp,
    
    -- Location and metadata
    location text,
    metadata text, -- JSON string for Dict[str, Any]
    
    -- Record metadata
    created_at timestamp,
    
    -- Primary key definition
    PRIMARY KEY (machine_id, timestamp)
) WITH CLUSTERING ORDER BY (timestamp DESC);

-- Create indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_machine_status_operator_id ON machine_status (operator_id);
CREATE INDEX IF NOT EXISTS idx_machine_status_status ON machine_status (status);
CREATE INDEX IF NOT EXISTS idx_machine_status_health_score ON machine_status (health_score);
CREATE INDEX IF NOT EXISTS idx_machine_status_maintenance_required ON machine_status (maintenance_required);

-- Create materialized view for operator-based queries
CREATE MATERIALIZED VIEW IF NOT EXISTS machine_status_by_operator AS
    SELECT operator_id, timestamp, machine_id, status, health_score, maintenance_required
    FROM machine_status
    WHERE operator_id IS NOT NULL AND timestamp IS NOT NULL AND machine_id IS NOT NULL
    PRIMARY KEY (operator_id, timestamp, machine_id)
    WITH CLUSTERING ORDER BY (timestamp DESC);

-- Create materialized view for status-based queries
CREATE MATERIALIZED VIEW IF NOT EXISTS machine_status_by_status AS
    SELECT status, timestamp, machine_id, operator_id, health_score, maintenance_required
    FROM machine_status
    WHERE status IS NOT NULL AND timestamp IS NOT NULL AND machine_id IS NOT NULL
    PRIMARY KEY (status, timestamp, machine_id)
    WITH CLUSTERING ORDER BY (timestamp DESC);

-- Create materialized view for maintenance queries
CREATE MATERIALIZED VIEW IF NOT EXISTS machine_status_maintenance AS
    SELECT maintenance_required, timestamp, machine_id, operator_id, status, health_score
    FROM machine_status
    WHERE maintenance_required IS NOT NULL AND timestamp IS NOT NULL AND machine_id IS NOT NULL
    PRIMARY KEY (maintenance_required, timestamp, machine_id)
    WITH CLUSTERING ORDER BY (timestamp DESC);

-- Set table properties for time-series optimization
ALTER TABLE machine_status WITH
    compaction = {'class': 'TimeWindowCompactionStrategy', 'compaction_window_unit': 'DAYS', 'compaction_window_size': 1}
    AND compression = {'sstable_compression': 'LZ4Compressor'}
    AND gc_grace_seconds = 864000; -- 10 days
