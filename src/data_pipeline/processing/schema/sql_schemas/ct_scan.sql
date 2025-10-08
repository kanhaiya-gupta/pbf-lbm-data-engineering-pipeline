-- CT Scan Data Table Schema
-- This table stores CT scan data for PBF-LB/M additive manufacturing quality assessment

CREATE TABLE IF NOT EXISTS ct_scan_data (
    -- Primary key and identifiers
    scan_id VARCHAR(100) PRIMARY KEY,
    process_id VARCHAR(100) NOT NULL,
    part_id VARCHAR(100),
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Scan information
    scan_type VARCHAR(30) NOT NULL CHECK (scan_type IN ('QUALITY_CONTROL', 'DEFECT_ANALYSIS', 'DIMENSIONAL_MEASUREMENT', 'MATERIAL_ANALYSIS', 'RESEARCH')),
    processing_status VARCHAR(20) NOT NULL CHECK (processing_status IN ('PENDING', 'IN_PROGRESS', 'COMPLETED', 'FAILED', 'CANCELLED')),
    
    -- Scan parameters
    voltage DECIMAL(8,2) NOT NULL CHECK (voltage >= 10 AND voltage <= 500),
    current DECIMAL(8,2) NOT NULL CHECK (current >= 0.1 AND current <= 1000),
    exposure_time DECIMAL(8,3) NOT NULL CHECK (exposure_time >= 0.001 AND exposure_time <= 60),
    number_of_projections INTEGER NOT NULL CHECK (number_of_projections >= 100 AND number_of_projections <= 10000),
    detector_resolution VARCHAR(20) NOT NULL,
    voxel_size DECIMAL(8,2) NOT NULL CHECK (voxel_size >= 0.1 AND voxel_size <= 1000),
    scan_duration DECIMAL(8,2) NOT NULL CHECK (scan_duration >= 0.1 AND scan_duration <= 1440),
    
    -- File metadata
    file_path VARCHAR(500) NOT NULL,
    file_format VARCHAR(10) NOT NULL CHECK (file_format IN ('DICOM', 'TIFF', 'RAW', 'NIFTI', 'MHD')),
    file_size BIGINT NOT NULL CHECK (file_size >= 0),
    compression VARCHAR(10) CHECK (compression IN ('GZIP', 'LZ4', 'ZSTD', 'BZIP2')),
    checksum VARCHAR(64),
    
    -- Image dimensions
    image_width INTEGER NOT NULL CHECK (image_width >= 1 AND image_width <= 10000),
    image_height INTEGER NOT NULL CHECK (image_height >= 1 AND image_height <= 10000),
    image_depth INTEGER NOT NULL CHECK (image_depth >= 1 AND image_depth <= 10000),
    physical_width DECIMAL(10,3) NOT NULL CHECK (physical_width >= 0.1 AND physical_width <= 1000),
    physical_height DECIMAL(10,3) NOT NULL CHECK (physical_height >= 0.1 AND physical_height <= 1000),
    physical_depth DECIMAL(10,3) NOT NULL CHECK (physical_depth >= 0.1 AND physical_depth <= 1000),
    
    -- Quality metrics
    contrast_to_noise_ratio DECIMAL(10,4),
    signal_to_noise_ratio DECIMAL(10,4),
    spatial_resolution DECIMAL(10,4),
    uniformity DECIMAL(5,2) CHECK (uniformity >= 0 AND uniformity <= 100),
    artifacts_detected BOOLEAN,
    artifact_severity VARCHAR(10) CHECK (artifact_severity IN ('NONE', 'MINIMAL', 'MODERATE', 'SEVERE')),
    
    -- Defect analysis
    total_defects INTEGER CHECK (total_defects >= 0),
    overall_quality_score DECIMAL(5,2) CHECK (overall_quality_score >= 0 AND overall_quality_score <= 100),
    acceptance_status VARCHAR(20) CHECK (acceptance_status IN ('ACCEPTED', 'REJECTED', 'CONDITIONAL', 'REQUIRES_REVIEW')),
    
    -- Dimensional analysis
    dimensional_accuracy DECIMAL(5,2) CHECK (dimensional_accuracy >= 0 AND dimensional_accuracy <= 100),
    
    -- Additional data
    processing_metadata JSONB,
    metadata JSONB
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_ct_scan_created_at ON ct_scan_data (created_at);
CREATE INDEX IF NOT EXISTS idx_ct_scan_process_id ON ct_scan_data (process_id);
CREATE INDEX IF NOT EXISTS idx_ct_scan_part_id ON ct_scan_data (part_id);
CREATE INDEX IF NOT EXISTS idx_ct_scan_scan_type ON ct_scan_data (scan_type);
CREATE INDEX IF NOT EXISTS idx_ct_scan_processing_status ON ct_scan_data (processing_status);
CREATE INDEX IF NOT EXISTS idx_ct_scan_file_format ON ct_scan_data (file_format);
CREATE INDEX IF NOT EXISTS idx_ct_scan_acceptance_status ON ct_scan_data (acceptance_status);
CREATE INDEX IF NOT EXISTS idx_ct_scan_overall_quality_score ON ct_scan_data (overall_quality_score);
CREATE INDEX IF NOT EXISTS idx_ct_scan_total_defects ON ct_scan_data (total_defects);

-- Create composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_ct_scan_process_created ON ct_scan_data (process_id, created_at);
CREATE INDEX IF NOT EXISTS idx_ct_scan_part_created ON ct_scan_data (part_id, created_at);
CREATE INDEX IF NOT EXISTS idx_ct_scan_type_status ON ct_scan_data (scan_type, processing_status);

-- Create partial indexes for quality analysis
CREATE INDEX IF NOT EXISTS idx_ct_scan_quality_issues ON ct_scan_data (created_at, overall_quality_score) WHERE overall_quality_score < 80;
CREATE INDEX IF NOT EXISTS idx_ct_scan_defects ON ct_scan_data (created_at, total_defects) WHERE total_defects > 0;
CREATE INDEX IF NOT EXISTS idx_ct_scan_rejected ON ct_scan_data (created_at, process_id) WHERE acceptance_status = 'REJECTED';

-- Create GIN index for JSONB columns
CREATE INDEX IF NOT EXISTS idx_ct_scan_processing_metadata_gin ON ct_scan_data USING GIN (processing_metadata);
CREATE INDEX IF NOT EXISTS idx_ct_scan_metadata_gin ON ct_scan_data USING GIN (metadata);

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_ct_scan_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_ct_scan_updated_at
    BEFORE UPDATE ON ct_scan_data
    FOR EACH ROW
    EXECUTE FUNCTION update_ct_scan_updated_at();

-- Create table for defect types
CREATE TABLE IF NOT EXISTS ct_scan_defect_types (
    id SERIAL PRIMARY KEY,
    scan_id VARCHAR(100) NOT NULL REFERENCES ct_scan_data(scan_id) ON DELETE CASCADE,
    defect_type VARCHAR(30) NOT NULL CHECK (defect_type IN ('POROSITY', 'CRACK', 'INCLUSION', 'DELAMINATION', 'WARPAGE', 'SHRINKAGE', 'UNMELTED_POWDER')),
    defect_count INTEGER NOT NULL CHECK (defect_count >= 0),
    average_size DECIMAL(10,3) NOT NULL CHECK (average_size >= 0),
    max_size DECIMAL(10,3) NOT NULL CHECK (max_size >= 0),
    severity VARCHAR(10) NOT NULL CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for defect types table
CREATE INDEX IF NOT EXISTS idx_ct_scan_defect_types_scan_id ON ct_scan_defect_types (scan_id);
CREATE INDEX IF NOT EXISTS idx_ct_scan_defect_types_type ON ct_scan_defect_types (defect_type);
CREATE INDEX IF NOT EXISTS idx_ct_scan_defect_types_severity ON ct_scan_defect_types (severity);

-- Create table for dimensional measurements
CREATE TABLE IF NOT EXISTS ct_scan_dimensional_measurements (
    id SERIAL PRIMARY KEY,
    scan_id VARCHAR(100) NOT NULL REFERENCES ct_scan_data(scan_id) ON DELETE CASCADE,
    dimension_name VARCHAR(100) NOT NULL,
    measured_value DECIMAL(10,3) NOT NULL,
    tolerance_deviation DECIMAL(10,3),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for dimensional measurements table
CREATE INDEX IF NOT EXISTS idx_ct_scan_dimensional_measurements_scan_id ON ct_scan_dimensional_measurements (scan_id);
CREATE INDEX IF NOT EXISTS idx_ct_scan_dimensional_measurements_dimension ON ct_scan_dimensional_measurements (dimension_name);

-- Create view for scan summary
CREATE OR REPLACE VIEW ct_scan_summary AS
SELECT 
    process_id,
    part_id,
    scan_type,
    COUNT(*) as total_scans,
    MIN(created_at) as first_scan,
    MAX(created_at) as last_scan,
    COUNT(CASE WHEN processing_status = 'COMPLETED' THEN 1 END) as completed_scans,
    COUNT(CASE WHEN processing_status = 'FAILED' THEN 1 END) as failed_scans,
    AVG(overall_quality_score) as avg_quality_score,
    SUM(total_defects) as total_defects,
    COUNT(CASE WHEN acceptance_status = 'ACCEPTED' THEN 1 END) as accepted_scans,
    COUNT(CASE WHEN acceptance_status = 'REJECTED' THEN 1 END) as rejected_scans
FROM ct_scan_data
GROUP BY process_id, part_id, scan_type;

-- Create view for defect analysis
CREATE OR REPLACE VIEW ct_scan_defect_analysis AS
SELECT 
    csd.scan_id,
    csd.process_id,
    csd.part_id,
    csd.scan_type,
    csd.created_at,
    csd.total_defects,
    csd.overall_quality_score,
    csd.acceptance_status,
    COUNT(dt.id) as defect_type_count,
    STRING_AGG(DISTINCT dt.defect_type, ', ') as defect_types,
    MAX(dt.severity) as max_severity,
    SUM(dt.defect_count) as total_defect_count
FROM ct_scan_data csd
LEFT JOIN ct_scan_defect_types dt ON csd.scan_id = dt.scan_id
GROUP BY csd.scan_id, csd.process_id, csd.part_id, csd.scan_type, csd.created_at, 
         csd.total_defects, csd.overall_quality_score, csd.acceptance_status;

-- Create view for quality trends
CREATE OR REPLACE VIEW ct_scan_quality_trends AS
SELECT 
    process_id,
    part_id,
    DATE_TRUNC('day', created_at) as date_bucket,
    COUNT(*) as scans_per_day,
    AVG(overall_quality_score) as avg_quality_score,
    AVG(total_defects) as avg_defects,
    COUNT(CASE WHEN acceptance_status = 'ACCEPTED' THEN 1 END) as accepted_count,
    COUNT(CASE WHEN acceptance_status = 'REJECTED' THEN 1 END) as rejected_count,
    COUNT(CASE WHEN artifacts_detected = true THEN 1 END) as artifacts_count
FROM ct_scan_data
WHERE processing_status = 'COMPLETED'
GROUP BY process_id, part_id, date_bucket
ORDER BY process_id, part_id, date_bucket;

-- Create table for scan statistics
CREATE TABLE IF NOT EXISTS ct_scan_statistics (
    id SERIAL PRIMARY KEY,
    process_id VARCHAR(100) NOT NULL,
    part_id VARCHAR(100),
    date DATE NOT NULL,
    total_scans INTEGER NOT NULL,
    completed_scans INTEGER DEFAULT 0,
    failed_scans INTEGER DEFAULT 0,
    avg_quality_score DECIMAL(5,2),
    total_defects INTEGER DEFAULT 0,
    accepted_scans INTEGER DEFAULT 0,
    rejected_scans INTEGER DEFAULT 0,
    avg_scan_duration DECIMAL(8,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(process_id, part_id, date)
);

-- Create index on statistics table
CREATE INDEX IF NOT EXISTS idx_ct_scan_statistics_process_date ON ct_scan_statistics (process_id, date);
CREATE INDEX IF NOT EXISTS idx_ct_scan_statistics_part_date ON ct_scan_statistics (part_id, date);

-- Create function to calculate scan statistics
CREATE OR REPLACE FUNCTION calculate_ct_scan_statistics(
    p_process_id VARCHAR(100),
    p_part_id VARCHAR(100),
    p_date DATE
)
RETURNS VOID AS $$
DECLARE
    v_total_scans INTEGER;
    v_completed_scans INTEGER;
    v_failed_scans INTEGER;
    v_avg_quality_score DECIMAL(5,2);
    v_total_defects INTEGER;
    v_accepted_scans INTEGER;
    v_rejected_scans INTEGER;
    v_avg_scan_duration DECIMAL(8,2);
BEGIN
    SELECT 
        COUNT(*),
        COUNT(CASE WHEN processing_status = 'COMPLETED' THEN 1 END),
        COUNT(CASE WHEN processing_status = 'FAILED' THEN 1 END),
        AVG(overall_quality_score),
        SUM(total_defects),
        COUNT(CASE WHEN acceptance_status = 'ACCEPTED' THEN 1 END),
        COUNT(CASE WHEN acceptance_status = 'REJECTED' THEN 1 END),
        AVG(scan_duration)
    INTO 
        v_total_scans,
        v_completed_scans,
        v_failed_scans,
        v_avg_quality_score,
        v_total_defects,
        v_accepted_scans,
        v_rejected_scans,
        v_avg_scan_duration
    FROM ct_scan_data
    WHERE process_id = p_process_id
        AND (p_part_id IS NULL OR part_id = p_part_id)
        AND DATE(created_at) = p_date;
    
    -- Insert or update statistics
    INSERT INTO ct_scan_statistics (
        process_id, part_id, date, total_scans, completed_scans, failed_scans,
        avg_quality_score, total_defects, accepted_scans, rejected_scans, avg_scan_duration
    ) VALUES (
        p_process_id, p_part_id, p_date, v_total_scans, v_completed_scans, v_failed_scans,
        v_avg_quality_score, v_total_defects, v_accepted_scans, v_rejected_scans, v_avg_scan_duration
    )
    ON CONFLICT (process_id, part_id, date) 
    DO UPDATE SET
        total_scans = EXCLUDED.total_scans,
        completed_scans = EXCLUDED.completed_scans,
        failed_scans = EXCLUDED.failed_scans,
        avg_quality_score = EXCLUDED.avg_quality_score,
        total_defects = EXCLUDED.total_defects,
        accepted_scans = EXCLUDED.accepted_scans,
        rejected_scans = EXCLUDED.rejected_scans,
        avg_scan_duration = EXCLUDED.avg_scan_duration;
END;
$$ LANGUAGE plpgsql;

-- Create function to assess scan quality
CREATE OR REPLACE FUNCTION assess_ct_scan_quality(
    p_scan_id VARCHAR(100)
)
RETURNS TABLE (
    quality_grade VARCHAR(20),
    recommendations TEXT[]
) AS $$
DECLARE
    v_quality_score DECIMAL(5,2);
    v_total_defects INTEGER;
    v_artifacts_detected BOOLEAN;
    v_artifact_severity VARCHAR(10);
    v_recommendations TEXT[];
BEGIN
    -- Get scan data
    SELECT overall_quality_score, total_defects, artifacts_detected, artifact_severity
    INTO v_quality_score, v_total_defects, v_artifacts_detected, v_artifact_severity
    FROM ct_scan_data
    WHERE scan_id = p_scan_id;
    
    -- Initialize recommendations array
    v_recommendations := ARRAY[]::TEXT[];
    
    -- Assess quality grade
    IF v_quality_score >= 90 AND v_total_defects = 0 AND (v_artifacts_detected = false OR v_artifact_severity = 'NONE') THEN
        quality_grade := 'EXCELLENT';
    ELSIF v_quality_score >= 80 AND v_total_defects <= 5 AND (v_artifacts_detected = false OR v_artifact_severity IN ('NONE', 'MINIMAL')) THEN
        quality_grade := 'GOOD';
    ELSIF v_quality_score >= 70 AND v_total_defects <= 10 AND v_artifact_severity IN ('NONE', 'MINIMAL', 'MODERATE') THEN
        quality_grade := 'ACCEPTABLE';
    ELSE
        quality_grade := 'POOR';
    END IF;
    
    -- Generate recommendations
    IF v_quality_score < 80 THEN
        v_recommendations := array_append(v_recommendations, 'Improve scan parameters to increase quality score');
    END IF;
    
    IF v_total_defects > 5 THEN
        v_recommendations := array_append(v_recommendations, 'Review process parameters to reduce defects');
    END IF;
    
    IF v_artifacts_detected = true AND v_artifact_severity IN ('MODERATE', 'SEVERE') THEN
        v_recommendations := array_append(v_recommendations, 'Address scan artifacts by adjusting acquisition parameters');
    END IF;
    
    IF v_quality_score < 70 THEN
        v_recommendations := array_append(v_recommendations, 'Consider rescanning with improved parameters');
    END IF;
    
    RETURN QUERY SELECT quality_grade, v_recommendations;
END;
$$ LANGUAGE plpgsql;

-- Create comments for documentation
COMMENT ON TABLE ct_scan_data IS 'CT scan data for PBF-LB/M additive manufacturing quality assessment';
COMMENT ON COLUMN ct_scan_data.scan_id IS 'Unique identifier for the CT scan';
COMMENT ON COLUMN ct_scan_data.process_id IS 'Associated PBF process identifier';
COMMENT ON COLUMN ct_scan_data.part_id IS 'Manufactured part identifier';
COMMENT ON COLUMN ct_scan_data.scan_type IS 'Type of CT scan (QUALITY_CONTROL, DEFECT_ANALYSIS, etc.)';
COMMENT ON COLUMN ct_scan_data.processing_status IS 'CT scan processing status';
COMMENT ON COLUMN ct_scan_data.voltage IS 'X-ray tube voltage in kV';
COMMENT ON COLUMN ct_scan_data.current IS 'X-ray tube current in mA';
COMMENT ON COLUMN ct_scan_data.exposure_time IS 'Exposure time per projection in seconds';
COMMENT ON COLUMN ct_scan_data.number_of_projections IS 'Number of X-ray projections';
COMMENT ON COLUMN ct_scan_data.voxel_size IS 'Voxel size in micrometers';
COMMENT ON COLUMN ct_scan_data.scan_duration IS 'Total scan duration in minutes';
COMMENT ON COLUMN ct_scan_data.file_path IS 'Path to the CT scan file';
COMMENT ON COLUMN ct_scan_data.file_format IS 'File format of the CT scan';
COMMENT ON COLUMN ct_scan_data.file_size IS 'File size in bytes';
COMMENT ON COLUMN ct_scan_data.overall_quality_score IS 'Overall quality score (0-100)';
COMMENT ON COLUMN ct_scan_data.total_defects IS 'Total number of detected defects';
COMMENT ON COLUMN ct_scan_data.acceptance_status IS 'Part acceptance status based on CT analysis';
COMMENT ON COLUMN ct_scan_data.processing_metadata IS 'Processing metadata and parameters as JSON';
COMMENT ON COLUMN ct_scan_data.metadata IS 'Additional metadata as JSON';
