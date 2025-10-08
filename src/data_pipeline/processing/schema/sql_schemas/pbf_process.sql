-- PBF Process Data Table Schema
-- This table stores PBF-LB/M process data for additive manufacturing

CREATE TABLE IF NOT EXISTS pbf_process_data (
    -- Primary key and identifiers
    process_id VARCHAR(100) PRIMARY KEY,
    machine_id VARCHAR(50) NOT NULL,
    build_id VARCHAR(100),
    
    -- Timestamps
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Process parameters
    layer_number INTEGER CHECK (layer_number >= 0 AND layer_number <= 10000),
    temperature DECIMAL(10,2) NOT NULL CHECK (temperature >= 0 AND temperature <= 2000),
    pressure DECIMAL(10,2) NOT NULL CHECK (pressure >= 0 AND pressure <= 1000),
    laser_power DECIMAL(10,2) NOT NULL CHECK (laser_power >= 0 AND laser_power <= 1000),
    scan_speed DECIMAL(10,2) NOT NULL CHECK (scan_speed >= 0 AND scan_speed <= 10000),
    layer_height DECIMAL(8,3) NOT NULL CHECK (layer_height >= 0.01 AND layer_height <= 1.0),
    hatch_spacing DECIMAL(8,3) CHECK (hatch_spacing >= 0.01 AND hatch_spacing <= 1.0),
    exposure_time DECIMAL(8,2) CHECK (exposure_time >= 0 AND exposure_time <= 3600),
    
    -- Material and environment
    atmosphere VARCHAR(20) CHECK (atmosphere IN ('argon', 'nitrogen', 'helium', 'vacuum', 'air')),
    powder_material VARCHAR(100),
    powder_batch_id VARCHAR(100),
    
    -- Quality metrics
    density DECIMAL(5,2) CHECK (density >= 0 AND density <= 100),
    surface_roughness DECIMAL(8,2) CHECK (surface_roughness >= 0 AND surface_roughness <= 100),
    dimensional_accuracy DECIMAL(8,3) CHECK (dimensional_accuracy >= 0 AND dimensional_accuracy <= 10),
    defect_count INTEGER CHECK (defect_count >= 0 AND defect_count <= 10000),
    
    -- Additional data
    process_parameters JSONB,
    metadata JSONB
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_pbf_process_timestamp ON pbf_process_data (timestamp);
CREATE INDEX IF NOT EXISTS idx_pbf_process_machine_id ON pbf_process_data (machine_id);
CREATE INDEX IF NOT EXISTS idx_pbf_process_build_id ON pbf_process_data (build_id);
CREATE INDEX IF NOT EXISTS idx_pbf_process_layer_number ON pbf_process_data (layer_number);
CREATE INDEX IF NOT EXISTS idx_pbf_process_temperature ON pbf_process_data (temperature);
CREATE INDEX IF NOT EXISTS idx_pbf_process_pressure ON pbf_process_data (pressure);
CREATE INDEX IF NOT EXISTS idx_pbf_process_laser_power ON pbf_process_data (laser_power);
CREATE INDEX IF NOT EXISTS idx_pbf_process_scan_speed ON pbf_process_data (scan_speed);
CREATE INDEX IF NOT EXISTS idx_pbf_process_created_at ON pbf_process_data (created_at);

-- Create composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_pbf_process_machine_timestamp ON pbf_process_data (machine_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_pbf_process_build_layer ON pbf_process_data (build_id, layer_number);

-- Create partial indexes for quality metrics
CREATE INDEX IF NOT EXISTS idx_pbf_process_defect_count ON pbf_process_data (defect_count) WHERE defect_count > 0;
CREATE INDEX IF NOT EXISTS idx_pbf_process_density ON pbf_process_data (density) WHERE density IS NOT NULL;

-- Create GIN index for JSONB columns
CREATE INDEX IF NOT EXISTS idx_pbf_process_parameters_gin ON pbf_process_data USING GIN (process_parameters);
CREATE INDEX IF NOT EXISTS idx_pbf_process_metadata_gin ON pbf_process_data USING GIN (metadata);

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_pbf_process_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_pbf_process_updated_at
    BEFORE UPDATE ON pbf_process_data
    FOR EACH ROW
    EXECUTE FUNCTION update_pbf_process_updated_at();

-- Create view for process summary
CREATE OR REPLACE VIEW pbf_process_summary AS
SELECT 
    machine_id,
    build_id,
    COUNT(*) as total_records,
    MIN(timestamp) as first_timestamp,
    MAX(timestamp) as last_timestamp,
    MIN(layer_number) as min_layer,
    MAX(layer_number) as max_layer,
    AVG(temperature) as avg_temperature,
    AVG(pressure) as avg_pressure,
    AVG(laser_power) as avg_laser_power,
    AVG(scan_speed) as avg_scan_speed,
    AVG(density) as avg_density,
    SUM(defect_count) as total_defects
FROM pbf_process_data
GROUP BY machine_id, build_id;

-- Create view for quality metrics
CREATE OR REPLACE VIEW pbf_process_quality_metrics AS
SELECT 
    process_id,
    machine_id,
    build_id,
    timestamp,
    layer_number,
    density,
    surface_roughness,
    dimensional_accuracy,
    defect_count,
    CASE 
        WHEN density >= 95 AND surface_roughness <= 10 AND dimensional_accuracy <= 0.1 AND defect_count = 0 THEN 'EXCELLENT'
        WHEN density >= 90 AND surface_roughness <= 20 AND dimensional_accuracy <= 0.2 AND defect_count <= 5 THEN 'GOOD'
        WHEN density >= 85 AND surface_roughness <= 30 AND dimensional_accuracy <= 0.5 AND defect_count <= 10 THEN 'ACCEPTABLE'
        ELSE 'POOR'
    END as quality_grade
FROM pbf_process_data
WHERE density IS NOT NULL;

-- Create table for process statistics
CREATE TABLE IF NOT EXISTS pbf_process_statistics (
    id SERIAL PRIMARY KEY,
    machine_id VARCHAR(50) NOT NULL,
    build_id VARCHAR(100),
    date DATE NOT NULL,
    total_records INTEGER NOT NULL,
    avg_temperature DECIMAL(10,2),
    avg_pressure DECIMAL(10,2),
    avg_laser_power DECIMAL(10,2),
    avg_scan_speed DECIMAL(10,2),
    avg_density DECIMAL(5,2),
    total_defects INTEGER,
    quality_grade VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(machine_id, build_id, date)
);

-- Create index on statistics table
CREATE INDEX IF NOT EXISTS idx_pbf_process_statistics_machine_date ON pbf_process_statistics (machine_id, date);
CREATE INDEX IF NOT EXISTS idx_pbf_process_statistics_build_date ON pbf_process_statistics (build_id, date);

-- Create function to calculate process statistics
CREATE OR REPLACE FUNCTION calculate_pbf_process_statistics(
    p_machine_id VARCHAR(50),
    p_build_id VARCHAR(100),
    p_date DATE
)
RETURNS VOID AS $$
DECLARE
    v_total_records INTEGER;
    v_avg_temperature DECIMAL(10,2);
    v_avg_pressure DECIMAL(10,2);
    v_avg_laser_power DECIMAL(10,2);
    v_avg_scan_speed DECIMAL(10,2);
    v_avg_density DECIMAL(5,2);
    v_total_defects INTEGER;
    v_quality_grade VARCHAR(20);
BEGIN
    SELECT 
        COUNT(*),
        AVG(temperature),
        AVG(pressure),
        AVG(laser_power),
        AVG(scan_speed),
        AVG(density),
        SUM(defect_count)
    INTO 
        v_total_records,
        v_avg_temperature,
        v_avg_pressure,
        v_avg_laser_power,
        v_avg_scan_speed,
        v_avg_density,
        v_total_defects
    FROM pbf_process_data
    WHERE machine_id = p_machine_id
        AND (p_build_id IS NULL OR build_id = p_build_id)
        AND DATE(timestamp) = p_date;
    
    -- Calculate quality grade
    IF v_avg_density >= 95 AND v_total_defects = 0 THEN
        v_quality_grade := 'EXCELLENT';
    ELSIF v_avg_density >= 90 AND v_total_defects <= 5 THEN
        v_quality_grade := 'GOOD';
    ELSIF v_avg_density >= 85 AND v_total_defects <= 10 THEN
        v_quality_grade := 'ACCEPTABLE';
    ELSE
        v_quality_grade := 'POOR';
    END IF;
    
    -- Insert or update statistics
    INSERT INTO pbf_process_statistics (
        machine_id, build_id, date, total_records, avg_temperature, 
        avg_pressure, avg_laser_power, avg_scan_speed, avg_density, 
        total_defects, quality_grade
    ) VALUES (
        p_machine_id, p_build_id, p_date, v_total_records, v_avg_temperature,
        v_avg_pressure, v_avg_laser_power, v_avg_scan_speed, v_avg_density,
        v_total_defects, v_quality_grade
    )
    ON CONFLICT (machine_id, build_id, date) 
    DO UPDATE SET
        total_records = EXCLUDED.total_records,
        avg_temperature = EXCLUDED.avg_temperature,
        avg_pressure = EXCLUDED.avg_pressure,
        avg_laser_power = EXCLUDED.avg_laser_power,
        avg_scan_speed = EXCLUDED.avg_scan_speed,
        avg_density = EXCLUDED.avg_density,
        total_defects = EXCLUDED.total_defects,
        quality_grade = EXCLUDED.quality_grade;
END;
$$ LANGUAGE plpgsql;

-- Create comments for documentation
COMMENT ON TABLE pbf_process_data IS 'PBF-LB/M process data for additive manufacturing';
COMMENT ON COLUMN pbf_process_data.process_id IS 'Unique identifier for the PBF process';
COMMENT ON COLUMN pbf_process_data.machine_id IS 'Identifier of the PBF machine';
COMMENT ON COLUMN pbf_process_data.build_id IS 'Build identifier for the manufacturing job';
COMMENT ON COLUMN pbf_process_data.timestamp IS 'Process timestamp';
COMMENT ON COLUMN pbf_process_data.layer_number IS 'Current layer number being processed';
COMMENT ON COLUMN pbf_process_data.temperature IS 'Process temperature in Celsius';
COMMENT ON COLUMN pbf_process_data.pressure IS 'Chamber pressure in mbar';
COMMENT ON COLUMN pbf_process_data.laser_power IS 'Laser power in watts';
COMMENT ON COLUMN pbf_process_data.scan_speed IS 'Laser scan speed in mm/s';
COMMENT ON COLUMN pbf_process_data.layer_height IS 'Layer height in mm';
COMMENT ON COLUMN pbf_process_data.density IS 'Part density percentage';
COMMENT ON COLUMN pbf_process_data.surface_roughness IS 'Surface roughness in micrometers';
COMMENT ON COLUMN pbf_process_data.dimensional_accuracy IS 'Dimensional accuracy in mm';
COMMENT ON COLUMN pbf_process_data.defect_count IS 'Number of detected defects';
COMMENT ON COLUMN pbf_process_data.process_parameters IS 'Additional process parameters as JSON';
COMMENT ON COLUMN pbf_process_data.metadata IS 'Additional metadata as JSON';
