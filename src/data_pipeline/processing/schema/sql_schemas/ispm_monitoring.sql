-- ISPM Monitoring Data Table Schema
-- This table stores ISPM (In-Situ Process Monitoring) data for PBF-LB/M additive manufacturing

CREATE TABLE IF NOT EXISTS ispm_monitoring_data (
    -- Primary key and identifiers
    monitoring_id VARCHAR(100) PRIMARY KEY,
    process_id VARCHAR(100) NOT NULL,
    sensor_id VARCHAR(50) NOT NULL,
    
    -- Timestamps
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Sensor information
    sensor_type VARCHAR(20) NOT NULL CHECK (sensor_type IN ('THERMAL', 'OPTICAL', 'ACOUSTIC', 'VIBRATION', 'PRESSURE', 'GAS_ANALYSIS', 'MELT_POOL', 'LAYER_HEIGHT')),
    sensor_location_x DECIMAL(10,3),
    sensor_location_y DECIMAL(10,3),
    sensor_location_z DECIMAL(10,3),
    
    -- Measurement data
    measurement_value DECIMAL(15,6) NOT NULL,
    unit VARCHAR(20) NOT NULL,
    measurement_range_min DECIMAL(15,6),
    measurement_range_max DECIMAL(15,6),
    measurement_accuracy DECIMAL(10,6),
    sampling_rate DECIMAL(10,2),
    
    -- Signal quality
    signal_quality VARCHAR(10) CHECK (signal_quality IN ('EXCELLENT', 'GOOD', 'FAIR', 'POOR', 'UNKNOWN')),
    noise_level DECIMAL(10,6),
    
    -- Calibration information
    calibration_status BOOLEAN,
    last_calibration_date TIMESTAMP WITH TIME ZONE,
    
    -- Environmental conditions
    ambient_temperature DECIMAL(8,2),
    relative_humidity DECIMAL(5,2) CHECK (relative_humidity >= 0 AND relative_humidity <= 100),
    vibration_level DECIMAL(10,6),
    
    -- Anomaly detection
    anomaly_detected BOOLEAN,
    anomaly_type VARCHAR(100),
    anomaly_severity VARCHAR(10) CHECK (anomaly_severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    
    -- Additional data
    raw_data BYTEA,
    processed_data JSONB,
    metadata JSONB
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_ispm_monitoring_timestamp ON ispm_monitoring_data (timestamp);
CREATE INDEX IF NOT EXISTS idx_ispm_monitoring_process_id ON ispm_monitoring_data (process_id);
CREATE INDEX IF NOT EXISTS idx_ispm_monitoring_sensor_id ON ispm_monitoring_data (sensor_id);
CREATE INDEX IF NOT EXISTS idx_ispm_monitoring_sensor_type ON ispm_monitoring_data (sensor_type);
CREATE INDEX IF NOT EXISTS idx_ispm_monitoring_measurement_value ON ispm_monitoring_data (measurement_value);
CREATE INDEX IF NOT EXISTS idx_ispm_monitoring_anomaly_detected ON ispm_monitoring_data (anomaly_detected);
CREATE INDEX IF NOT EXISTS idx_ispm_monitoring_anomaly_severity ON ispm_monitoring_data (anomaly_severity);
CREATE INDEX IF NOT EXISTS idx_ispm_monitoring_created_at ON ispm_monitoring_data (created_at);

-- Create composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_ispm_monitoring_process_timestamp ON ispm_monitoring_data (process_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_ispm_monitoring_sensor_timestamp ON ispm_monitoring_data (sensor_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_ispm_monitoring_type_timestamp ON ispm_monitoring_data (sensor_type, timestamp);

-- Create partial indexes for anomaly detection
CREATE INDEX IF NOT EXISTS idx_ispm_monitoring_anomalies ON ispm_monitoring_data (timestamp, anomaly_type, anomaly_severity) WHERE anomaly_detected = true;
CREATE INDEX IF NOT EXISTS idx_ispm_monitoring_critical_anomalies ON ispm_monitoring_data (timestamp, process_id) WHERE anomaly_severity = 'CRITICAL';

-- Create GIN index for JSONB columns
CREATE INDEX IF NOT EXISTS idx_ispm_monitoring_processed_data_gin ON ispm_monitoring_data USING GIN (processed_data);
CREATE INDEX IF NOT EXISTS idx_ispm_monitoring_metadata_gin ON ispm_monitoring_data USING GIN (metadata);

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_ispm_monitoring_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_ispm_monitoring_updated_at
    BEFORE UPDATE ON ispm_monitoring_data
    FOR EACH ROW
    EXECUTE FUNCTION update_ispm_monitoring_updated_at();

-- Create view for sensor summary
CREATE OR REPLACE VIEW ispm_sensor_summary AS
SELECT 
    sensor_id,
    sensor_type,
    COUNT(*) as total_measurements,
    MIN(timestamp) as first_measurement,
    MAX(timestamp) as last_measurement,
    AVG(measurement_value) as avg_measurement_value,
    MIN(measurement_value) as min_measurement_value,
    MAX(measurement_value) as max_measurement_value,
    STDDEV(measurement_value) as stddev_measurement_value,
    COUNT(CASE WHEN anomaly_detected = true THEN 1 END) as anomaly_count,
    COUNT(CASE WHEN signal_quality = 'EXCELLENT' THEN 1 END) as excellent_signal_count,
    COUNT(CASE WHEN signal_quality = 'GOOD' THEN 1 END) as good_signal_count,
    COUNT(CASE WHEN signal_quality = 'FAIR' THEN 1 END) as fair_signal_count,
    COUNT(CASE WHEN signal_quality = 'POOR' THEN 1 END) as poor_signal_count
FROM ispm_monitoring_data
GROUP BY sensor_id, sensor_type;

-- Create view for anomaly analysis
CREATE OR REPLACE VIEW ispm_anomaly_analysis AS
SELECT 
    process_id,
    sensor_type,
    anomaly_type,
    anomaly_severity,
    COUNT(*) as anomaly_count,
    MIN(timestamp) as first_anomaly,
    MAX(timestamp) as last_anomaly,
    AVG(measurement_value) as avg_measurement_during_anomaly,
    AVG(noise_level) as avg_noise_level
FROM ispm_monitoring_data
WHERE anomaly_detected = true
GROUP BY process_id, sensor_type, anomaly_type, anomaly_severity;

-- Create view for signal quality trends
CREATE OR REPLACE VIEW ispm_signal_quality_trends AS
SELECT 
    sensor_id,
    sensor_type,
    DATE_TRUNC('hour', timestamp) as hour_bucket,
    COUNT(*) as total_measurements,
    COUNT(CASE WHEN signal_quality = 'EXCELLENT' THEN 1 END) as excellent_count,
    COUNT(CASE WHEN signal_quality = 'GOOD' THEN 1 END) as good_count,
    COUNT(CASE WHEN signal_quality = 'FAIR' THEN 1 END) as fair_count,
    COUNT(CASE WHEN signal_quality = 'POOR' THEN 1 END) as poor_count,
    AVG(noise_level) as avg_noise_level,
    AVG(measurement_accuracy) as avg_measurement_accuracy
FROM ispm_monitoring_data
GROUP BY sensor_id, sensor_type, hour_bucket
ORDER BY sensor_id, hour_bucket;

-- Create table for sensor statistics
CREATE TABLE IF NOT EXISTS ispm_sensor_statistics (
    id SERIAL PRIMARY KEY,
    sensor_id VARCHAR(50) NOT NULL,
    sensor_type VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    total_measurements INTEGER NOT NULL,
    avg_measurement_value DECIMAL(15,6),
    min_measurement_value DECIMAL(15,6),
    max_measurement_value DECIMAL(15,6),
    stddev_measurement_value DECIMAL(15,6),
    anomaly_count INTEGER DEFAULT 0,
    excellent_signal_count INTEGER DEFAULT 0,
    good_signal_count INTEGER DEFAULT 0,
    fair_signal_count INTEGER DEFAULT 0,
    poor_signal_count INTEGER DEFAULT 0,
    avg_noise_level DECIMAL(10,6),
    avg_measurement_accuracy DECIMAL(10,6),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(sensor_id, date)
);

-- Create index on statistics table
CREATE INDEX IF NOT EXISTS idx_ispm_sensor_statistics_sensor_date ON ispm_sensor_statistics (sensor_id, date);
CREATE INDEX IF NOT EXISTS idx_ispm_sensor_statistics_type_date ON ispm_sensor_statistics (sensor_type, date);

-- Create function to calculate sensor statistics
CREATE OR REPLACE FUNCTION calculate_ispm_sensor_statistics(
    p_sensor_id VARCHAR(50),
    p_date DATE
)
RETURNS VOID AS $$
DECLARE
    v_total_measurements INTEGER;
    v_avg_measurement_value DECIMAL(15,6);
    v_min_measurement_value DECIMAL(15,6);
    v_max_measurement_value DECIMAL(15,6);
    v_stddev_measurement_value DECIMAL(15,6);
    v_anomaly_count INTEGER;
    v_excellent_signal_count INTEGER;
    v_good_signal_count INTEGER;
    v_fair_signal_count INTEGER;
    v_poor_signal_count INTEGER;
    v_avg_noise_level DECIMAL(10,6);
    v_avg_measurement_accuracy DECIMAL(10,6);
    v_sensor_type VARCHAR(20);
BEGIN
    SELECT 
        COUNT(*),
        AVG(measurement_value),
        MIN(measurement_value),
        MAX(measurement_value),
        STDDEV(measurement_value),
        COUNT(CASE WHEN anomaly_detected = true THEN 1 END),
        COUNT(CASE WHEN signal_quality = 'EXCELLENT' THEN 1 END),
        COUNT(CASE WHEN signal_quality = 'GOOD' THEN 1 END),
        COUNT(CASE WHEN signal_quality = 'FAIR' THEN 1 END),
        COUNT(CASE WHEN signal_quality = 'POOR' THEN 1 END),
        AVG(noise_level),
        AVG(measurement_accuracy),
        MAX(sensor_type)
    INTO 
        v_total_measurements,
        v_avg_measurement_value,
        v_min_measurement_value,
        v_max_measurement_value,
        v_stddev_measurement_value,
        v_anomaly_count,
        v_excellent_signal_count,
        v_good_signal_count,
        v_fair_signal_count,
        v_poor_signal_count,
        v_avg_noise_level,
        v_avg_measurement_accuracy,
        v_sensor_type
    FROM ispm_monitoring_data
    WHERE sensor_id = p_sensor_id
        AND DATE(timestamp) = p_date;
    
    -- Insert or update statistics
    INSERT INTO ispm_sensor_statistics (
        sensor_id, sensor_type, date, total_measurements, avg_measurement_value,
        min_measurement_value, max_measurement_value, stddev_measurement_value,
        anomaly_count, excellent_signal_count, good_signal_count, fair_signal_count,
        poor_signal_count, avg_noise_level, avg_measurement_accuracy
    ) VALUES (
        p_sensor_id, v_sensor_type, p_date, v_total_measurements, v_avg_measurement_value,
        v_min_measurement_value, v_max_measurement_value, v_stddev_measurement_value,
        v_anomaly_count, v_excellent_signal_count, v_good_signal_count, v_fair_signal_count,
        v_poor_signal_count, v_avg_noise_level, v_avg_measurement_accuracy
    )
    ON CONFLICT (sensor_id, date) 
    DO UPDATE SET
        total_measurements = EXCLUDED.total_measurements,
        avg_measurement_value = EXCLUDED.avg_measurement_value,
        min_measurement_value = EXCLUDED.min_measurement_value,
        max_measurement_value = EXCLUDED.max_measurement_value,
        stddev_measurement_value = EXCLUDED.stddev_measurement_value,
        anomaly_count = EXCLUDED.anomaly_count,
        excellent_signal_count = EXCLUDED.excellent_signal_count,
        good_signal_count = EXCLUDED.good_signal_count,
        fair_signal_count = EXCLUDED.fair_signal_count,
        poor_signal_count = EXCLUDED.poor_signal_count,
        avg_noise_level = EXCLUDED.avg_noise_level,
        avg_measurement_accuracy = EXCLUDED.avg_measurement_accuracy;
END;
$$ LANGUAGE plpgsql;

-- Create function to detect anomalies
CREATE OR REPLACE FUNCTION detect_ispm_anomalies(
    p_sensor_id VARCHAR(50),
    p_threshold_multiplier DECIMAL DEFAULT 3.0
)
RETURNS TABLE (
    monitoring_id VARCHAR(100),
    "timestamp" TIMESTAMP WITH TIME ZONE,
    measurement_value DECIMAL(15,6),
    anomaly_type VARCHAR(100),
    anomaly_severity VARCHAR(10)
) AS $$
DECLARE
    v_avg_value DECIMAL(15,6);
    v_stddev_value DECIMAL(15,6);
    v_threshold DECIMAL(15,6);
BEGIN
    -- Calculate statistics for the sensor
    SELECT AVG(measurement_value), STDDEV(measurement_value)
    INTO v_avg_value, v_stddev_value
    FROM ispm_monitoring_data
    WHERE sensor_id = p_sensor_id
        AND timestamp >= CURRENT_TIMESTAMP - INTERVAL '24 hours';
    
    -- Calculate threshold
    v_threshold := v_stddev_value * p_threshold_multiplier;
    
    -- Return anomalies
    RETURN QUERY
    SELECT 
        imd.monitoring_id,
        imd.timestamp,
        imd.measurement_value,
        CASE 
            WHEN imd.measurement_value > v_avg_value + v_threshold THEN 'HIGH_VALUE'
            WHEN imd.measurement_value < v_avg_value - v_threshold THEN 'LOW_VALUE'
            ELSE 'UNKNOWN'
        END as anomaly_type,
        CASE 
            WHEN ABS(imd.measurement_value - v_avg_value) > v_threshold * 2 THEN 'CRITICAL'
            WHEN ABS(imd.measurement_value - v_avg_value) > v_threshold * 1.5 THEN 'HIGH'
            ELSE 'MEDIUM'
        END as anomaly_severity
    FROM ispm_monitoring_data imd
    WHERE imd.sensor_id = p_sensor_id
        AND imd.timestamp >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
        AND ABS(imd.measurement_value - v_avg_value) > v_threshold;
END;
$$ LANGUAGE plpgsql;

-- Create comments for documentation
COMMENT ON TABLE ispm_monitoring_data IS 'ISPM (In-Situ Process Monitoring) data for PBF-LB/M additive manufacturing';
COMMENT ON COLUMN ispm_monitoring_data.monitoring_id IS 'Unique identifier for the monitoring record';
COMMENT ON COLUMN ispm_monitoring_data.process_id IS 'Associated PBF process identifier';
COMMENT ON COLUMN ispm_monitoring_data.sensor_id IS 'Sensor identifier';
COMMENT ON COLUMN ispm_monitoring_data.sensor_type IS 'Type of sensor (THERMAL, OPTICAL, ACOUSTIC, etc.)';
COMMENT ON COLUMN ispm_monitoring_data.timestamp IS 'Monitoring timestamp';
COMMENT ON COLUMN ispm_monitoring_data.measurement_value IS 'Primary measurement value';
COMMENT ON COLUMN ispm_monitoring_data.unit IS 'Unit of measurement';
COMMENT ON COLUMN ispm_monitoring_data.signal_quality IS 'Signal quality assessment';
COMMENT ON COLUMN ispm_monitoring_data.noise_level IS 'Noise level in the signal';
COMMENT ON COLUMN ispm_monitoring_data.anomaly_detected IS 'Whether an anomaly was detected';
COMMENT ON COLUMN ispm_monitoring_data.anomaly_type IS 'Type of detected anomaly';
COMMENT ON COLUMN ispm_monitoring_data.anomaly_severity IS 'Severity of detected anomaly';
COMMENT ON COLUMN ispm_monitoring_data.processed_data IS 'Processed sensor data as JSON';
COMMENT ON COLUMN ispm_monitoring_data.metadata IS 'Additional metadata as JSON';
