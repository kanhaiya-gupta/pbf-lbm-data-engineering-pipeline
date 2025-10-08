-- Powder Bed Monitoring Data Table Schema
-- This table stores powder bed monitoring data for PBF-LB/M additive manufacturing

CREATE TABLE IF NOT EXISTS powder_bed_data (
    -- Primary key and identifiers
    bed_id VARCHAR(100) PRIMARY KEY,
    process_id VARCHAR(100) NOT NULL,
    layer_number INTEGER NOT NULL CHECK (layer_number >= 0 AND layer_number <= 10000),
    
    -- Timestamps
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Image metadata
    image_id VARCHAR(100) NOT NULL,
    camera_id VARCHAR(50) NOT NULL,
    image_format VARCHAR(10) NOT NULL CHECK (image_format IN ('JPEG', 'PNG', 'TIFF', 'RAW', 'BMP')),
    resolution VARCHAR(20) NOT NULL,
    file_size BIGINT NOT NULL CHECK (file_size >= 0),
    file_path VARCHAR(500) NOT NULL,
    
    -- Camera capture settings
    exposure_time DECIMAL(8,3) NOT NULL CHECK (exposure_time >= 0.001 AND exposure_time <= 60),
    aperture DECIMAL(4,1) NOT NULL CHECK (aperture >= 0.5 AND aperture <= 32),
    iso INTEGER NOT NULL CHECK (iso >= 50 AND iso <= 25600),
    white_balance VARCHAR(50) NOT NULL,
    lighting_conditions VARCHAR(100) NOT NULL,
    
    -- Powder characteristics
    material_type VARCHAR(100) NOT NULL,
    particle_size_d10 DECIMAL(8,2),
    particle_size_d50 DECIMAL(8,2),
    particle_size_d90 DECIMAL(8,2),
    particle_size_span DECIMAL(8,4),
    powder_density DECIMAL(6,3) CHECK (powder_density >= 0.1 AND powder_density <= 20),
    flowability DECIMAL(5,2) CHECK (flowability >= 0 AND flowability <= 100),
    moisture_content DECIMAL(5,2) CHECK (moisture_content >= 0 AND moisture_content <= 100),
    
    -- Bed quality metrics
    uniformity_score DECIMAL(5,2) NOT NULL CHECK (uniformity_score >= 0 AND uniformity_score <= 100),
    coverage_percentage DECIMAL(5,2) NOT NULL CHECK (coverage_percentage >= 0 AND coverage_percentage <= 100),
    thickness_consistency DECIMAL(5,2) NOT NULL CHECK (thickness_consistency >= 0 AND thickness_consistency <= 100),
    surface_roughness DECIMAL(8,2) CHECK (surface_roughness >= 0 AND surface_roughness <= 100),
    density_variation DECIMAL(6,4) CHECK (density_variation >= 0 AND density_variation <= 1),
    defect_density DECIMAL(10,4) CHECK (defect_density >= 0),
    
    -- Image analysis
    brightness DECIMAL(6,2) CHECK (brightness >= 0 AND brightness <= 255),
    contrast DECIMAL(10,4),
    sharpness DECIMAL(10,4),
    noise_level DECIMAL(10,4),
    red_channel DECIMAL(6,2) CHECK (red_channel >= 0 AND red_channel <= 255),
    green_channel DECIMAL(6,2) CHECK (green_channel >= 0 AND green_channel <= 255),
    blue_channel DECIMAL(6,2) CHECK (blue_channel >= 0 AND blue_channel <= 255),
    
    -- Texture analysis
    texture_homogeneity DECIMAL(6,4) CHECK (texture_homogeneity >= 0 AND texture_homogeneity <= 1),
    texture_contrast DECIMAL(10,4),
    texture_energy DECIMAL(6,4) CHECK (texture_energy >= 0 AND texture_energy <= 1),
    texture_entropy DECIMAL(10,4),
    
    -- Defect detection
    defects_detected BOOLEAN,
    defect_count INTEGER CHECK (defect_count >= 0),
    overall_quality_assessment VARCHAR(20) CHECK (overall_quality_assessment IN ('EXCELLENT', 'GOOD', 'ACCEPTABLE', 'POOR', 'UNACCEPTABLE')),
    
    -- Environmental conditions
    ambient_temperature DECIMAL(8,2) CHECK (ambient_temperature >= -50 AND ambient_temperature <= 100),
    relative_humidity DECIMAL(5,2) CHECK (relative_humidity >= 0 AND relative_humidity <= 100),
    atmospheric_pressure DECIMAL(8,2) CHECK (atmospheric_pressure >= 0 AND atmospheric_pressure <= 2000),
    vibration_level DECIMAL(10,6),
    
    -- Processing status
    processing_status VARCHAR(20) NOT NULL CHECK (processing_status IN ('PENDING', 'IN_PROGRESS', 'COMPLETED', 'FAILED', 'CANCELLED')),
    
    -- Additional data
    metadata JSONB
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_powder_bed_timestamp ON powder_bed_data (timestamp);
CREATE INDEX IF NOT EXISTS idx_powder_bed_process_id ON powder_bed_data (process_id);
CREATE INDEX IF NOT EXISTS idx_powder_bed_layer_number ON powder_bed_data (layer_number);
CREATE INDEX IF NOT EXISTS idx_powder_bed_camera_id ON powder_bed_data (camera_id);
CREATE INDEX IF NOT EXISTS idx_powder_bed_material_type ON powder_bed_data (material_type);
CREATE INDEX IF NOT EXISTS idx_powder_bed_uniformity_score ON powder_bed_data (uniformity_score);
CREATE INDEX IF NOT EXISTS idx_powder_bed_coverage_percentage ON powder_bed_data (coverage_percentage);
CREATE INDEX IF NOT EXISTS idx_powder_bed_defects_detected ON powder_bed_data (defects_detected);
CREATE INDEX IF NOT EXISTS idx_powder_bed_quality_assessment ON powder_bed_data (overall_quality_assessment);
CREATE INDEX IF NOT EXISTS idx_powder_bed_processing_status ON powder_bed_data (processing_status);
CREATE INDEX IF NOT EXISTS idx_powder_bed_created_at ON powder_bed_data (created_at);

-- Create composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_powder_bed_process_layer ON powder_bed_data (process_id, layer_number);
CREATE INDEX IF NOT EXISTS idx_powder_bed_process_timestamp ON powder_bed_data (process_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_powder_bed_layer_timestamp ON powder_bed_data (layer_number, timestamp);

-- Create partial indexes for quality analysis
CREATE INDEX IF NOT EXISTS idx_powder_bed_quality_issues ON powder_bed_data (timestamp, uniformity_score) WHERE uniformity_score < 70;
CREATE INDEX IF NOT EXISTS idx_powder_bed_defects ON powder_bed_data (timestamp, defect_count) WHERE defects_detected = true;
CREATE INDEX IF NOT EXISTS idx_powder_bed_poor_quality ON powder_bed_data (timestamp, process_id) WHERE overall_quality_assessment IN ('POOR', 'UNACCEPTABLE');

-- Create GIN index for JSONB columns
CREATE INDEX IF NOT EXISTS idx_powder_bed_metadata_gin ON powder_bed_data USING GIN (metadata);

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_powder_bed_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_powder_bed_updated_at
    BEFORE UPDATE ON powder_bed_data
    FOR EACH ROW
    EXECUTE FUNCTION update_powder_bed_updated_at();

-- Create table for powder bed defects
CREATE TABLE IF NOT EXISTS powder_bed_defects (
    id SERIAL PRIMARY KEY,
    bed_id VARCHAR(100) NOT NULL REFERENCES powder_bed_data(bed_id) ON DELETE CASCADE,
    defect_id VARCHAR(100) NOT NULL,
    defect_type VARCHAR(30) NOT NULL CHECK (defect_type IN ('INSUFFICIENT_POWDER', 'EXCESS_POWDER', 'CONTAMINATION', 'AGGLOMERATION', 'SEGREGATION', 'SURFACE_IRREGULARITY')),
    x_coordinate DECIMAL(10,3) NOT NULL,
    y_coordinate DECIMAL(10,3) NOT NULL,
    area DECIMAL(10,3) NOT NULL CHECK (area >= 0),
    severity VARCHAR(10) NOT NULL CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    confidence_score DECIMAL(5,2) NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for defects table
CREATE INDEX IF NOT EXISTS idx_powder_bed_defects_bed_id ON powder_bed_defects (bed_id);
CREATE INDEX IF NOT EXISTS idx_powder_bed_defects_type ON powder_bed_defects (defect_type);
CREATE INDEX IF NOT EXISTS idx_powder_bed_defects_severity ON powder_bed_defects (severity);
CREATE INDEX IF NOT EXISTS idx_powder_bed_defects_confidence ON powder_bed_defects (confidence_score);

-- Create view for powder bed summary
CREATE OR REPLACE VIEW powder_bed_summary AS
SELECT 
    process_id,
    material_type,
    COUNT(*) as total_layers,
    MIN(layer_number) as min_layer,
    MAX(layer_number) as max_layer,
    MIN(timestamp) as first_timestamp,
    MAX(timestamp) as last_timestamp,
    AVG(uniformity_score) as avg_uniformity_score,
    AVG(coverage_percentage) as avg_coverage_percentage,
    AVG(thickness_consistency) as avg_thickness_consistency,
    AVG(surface_roughness) as avg_surface_roughness,
    SUM(defect_count) as total_defects,
    COUNT(CASE WHEN defects_detected = true THEN 1 END) as layers_with_defects,
    COUNT(CASE WHEN overall_quality_assessment = 'EXCELLENT' THEN 1 END) as excellent_layers,
    COUNT(CASE WHEN overall_quality_assessment = 'GOOD' THEN 1 END) as good_layers,
    COUNT(CASE WHEN overall_quality_assessment = 'ACCEPTABLE' THEN 1 END) as acceptable_layers,
    COUNT(CASE WHEN overall_quality_assessment = 'POOR' THEN 1 END) as poor_layers,
    COUNT(CASE WHEN overall_quality_assessment = 'UNACCEPTABLE' THEN 1 END) as unacceptable_layers
FROM powder_bed_data
GROUP BY process_id, material_type;

-- Create view for defect analysis
CREATE OR REPLACE VIEW powder_bed_defect_analysis AS
SELECT 
    pbd.bed_id,
    pbd.process_id,
    pbd.layer_number,
    pbd.timestamp,
    pbd.material_type,
    pbd.defect_count,
    pbd.overall_quality_assessment,
    COUNT(pd.id) as detected_defect_count,
    STRING_AGG(DISTINCT pd.defect_type, ', ') as defect_types,
    MAX(pd.severity) as max_severity,
    AVG(pd.confidence_score) as avg_confidence_score,
    SUM(pd.area) as total_defect_area
FROM powder_bed_data pbd
LEFT JOIN powder_bed_defects pd ON pbd.bed_id = pd.bed_id
GROUP BY pbd.bed_id, pbd.process_id, pbd.layer_number, pbd.timestamp, 
         pbd.material_type, pbd.defect_count, pbd.overall_quality_assessment;

-- Create view for quality trends
CREATE OR REPLACE VIEW powder_bed_quality_trends AS
SELECT 
    process_id,
    material_type,
    DATE_TRUNC('hour', timestamp) as hour_bucket,
    COUNT(*) as layers_per_hour,
    AVG(uniformity_score) as avg_uniformity_score,
    AVG(coverage_percentage) as avg_coverage_percentage,
    AVG(thickness_consistency) as avg_thickness_consistency,
    AVG(surface_roughness) as avg_surface_roughness,
    SUM(defect_count) as total_defects,
    COUNT(CASE WHEN defects_detected = true THEN 1 END) as layers_with_defects,
    AVG(ambient_temperature) as avg_temperature,
    AVG(relative_humidity) as avg_humidity
FROM powder_bed_data
GROUP BY process_id, material_type, hour_bucket
ORDER BY process_id, material_type, hour_bucket;

-- Create table for powder bed statistics
CREATE TABLE IF NOT EXISTS powder_bed_statistics (
    id SERIAL PRIMARY KEY,
    process_id VARCHAR(100) NOT NULL,
    material_type VARCHAR(100) NOT NULL,
    date DATE NOT NULL,
    total_layers INTEGER NOT NULL,
    avg_uniformity_score DECIMAL(5,2),
    avg_coverage_percentage DECIMAL(5,2),
    avg_thickness_consistency DECIMAL(5,2),
    avg_surface_roughness DECIMAL(8,2),
    total_defects INTEGER DEFAULT 0,
    layers_with_defects INTEGER DEFAULT 0,
    excellent_layers INTEGER DEFAULT 0,
    good_layers INTEGER DEFAULT 0,
    acceptable_layers INTEGER DEFAULT 0,
    poor_layers INTEGER DEFAULT 0,
    unacceptable_layers INTEGER DEFAULT 0,
    avg_temperature DECIMAL(8,2),
    avg_humidity DECIMAL(5,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(process_id, material_type, date)
);

-- Create index on statistics table
CREATE INDEX IF NOT EXISTS idx_powder_bed_statistics_process_date ON powder_bed_statistics (process_id, date);
CREATE INDEX IF NOT EXISTS idx_powder_bed_statistics_material_date ON powder_bed_statistics (material_type, date);

-- Create function to calculate powder bed statistics
CREATE OR REPLACE FUNCTION calculate_powder_bed_statistics(
    p_process_id VARCHAR(100),
    p_material_type VARCHAR(100),
    p_date DATE
)
RETURNS VOID AS $$
DECLARE
    v_total_layers INTEGER;
    v_avg_uniformity_score DECIMAL(5,2);
    v_avg_coverage_percentage DECIMAL(5,2);
    v_avg_thickness_consistency DECIMAL(5,2);
    v_avg_surface_roughness DECIMAL(8,2);
    v_total_defects INTEGER;
    v_layers_with_defects INTEGER;
    v_excellent_layers INTEGER;
    v_good_layers INTEGER;
    v_acceptable_layers INTEGER;
    v_poor_layers INTEGER;
    v_unacceptable_layers INTEGER;
    v_avg_temperature DECIMAL(8,2);
    v_avg_humidity DECIMAL(5,2);
BEGIN
    SELECT 
        COUNT(*),
        AVG(uniformity_score),
        AVG(coverage_percentage),
        AVG(thickness_consistency),
        AVG(surface_roughness),
        SUM(defect_count),
        COUNT(CASE WHEN defects_detected = true THEN 1 END),
        COUNT(CASE WHEN overall_quality_assessment = 'EXCELLENT' THEN 1 END),
        COUNT(CASE WHEN overall_quality_assessment = 'GOOD' THEN 1 END),
        COUNT(CASE WHEN overall_quality_assessment = 'ACCEPTABLE' THEN 1 END),
        COUNT(CASE WHEN overall_quality_assessment = 'POOR' THEN 1 END),
        COUNT(CASE WHEN overall_quality_assessment = 'UNACCEPTABLE' THEN 1 END),
        AVG(ambient_temperature),
        AVG(relative_humidity)
    INTO 
        v_total_layers,
        v_avg_uniformity_score,
        v_avg_coverage_percentage,
        v_avg_thickness_consistency,
        v_avg_surface_roughness,
        v_total_defects,
        v_layers_with_defects,
        v_excellent_layers,
        v_good_layers,
        v_acceptable_layers,
        v_poor_layers,
        v_unacceptable_layers,
        v_avg_temperature,
        v_avg_humidity
    FROM powder_bed_data
    WHERE process_id = p_process_id
        AND material_type = p_material_type
        AND DATE(timestamp) = p_date;
    
    -- Insert or update statistics
    INSERT INTO powder_bed_statistics (
        process_id, material_type, date, total_layers, avg_uniformity_score,
        avg_coverage_percentage, avg_thickness_consistency, avg_surface_roughness,
        total_defects, layers_with_defects, excellent_layers, good_layers,
        acceptable_layers, poor_layers, unacceptable_layers, avg_temperature, avg_humidity
    ) VALUES (
        p_process_id, p_material_type, p_date, v_total_layers, v_avg_uniformity_score,
        v_avg_coverage_percentage, v_avg_thickness_consistency, v_avg_surface_roughness,
        v_total_defects, v_layers_with_defects, v_excellent_layers, v_good_layers,
        v_acceptable_layers, v_poor_layers, v_unacceptable_layers, v_avg_temperature, v_avg_humidity
    )
    ON CONFLICT (process_id, material_type, date) 
    DO UPDATE SET
        total_layers = EXCLUDED.total_layers,
        avg_uniformity_score = EXCLUDED.avg_uniformity_score,
        avg_coverage_percentage = EXCLUDED.avg_coverage_percentage,
        avg_thickness_consistency = EXCLUDED.avg_thickness_consistency,
        avg_surface_roughness = EXCLUDED.avg_surface_roughness,
        total_defects = EXCLUDED.total_defects,
        layers_with_defects = EXCLUDED.layers_with_defects,
        excellent_layers = EXCLUDED.excellent_layers,
        good_layers = EXCLUDED.good_layers,
        acceptable_layers = EXCLUDED.acceptable_layers,
        poor_layers = EXCLUDED.poor_layers,
        unacceptable_layers = EXCLUDED.unacceptable_layers,
        avg_temperature = EXCLUDED.avg_temperature,
        avg_humidity = EXCLUDED.avg_humidity;
END;
$$ LANGUAGE plpgsql;

-- Create function to assess powder bed quality
CREATE OR REPLACE FUNCTION assess_powder_bed_quality(
    p_bed_id VARCHAR(100)
)
RETURNS TABLE (
    quality_grade VARCHAR(20),
    recommendations TEXT[]
) AS $$
DECLARE
    v_uniformity_score DECIMAL(5,2);
    v_coverage_percentage DECIMAL(5,2);
    v_thickness_consistency DECIMAL(5,2);
    v_surface_roughness DECIMAL(8,2);
    v_defect_count INTEGER;
    v_recommendations TEXT[];
BEGIN
    -- Get powder bed data
    SELECT uniformity_score, coverage_percentage, thickness_consistency, 
           surface_roughness, defect_count
    INTO v_uniformity_score, v_coverage_percentage, v_thickness_consistency, 
         v_surface_roughness, v_defect_count
    FROM powder_bed_data
    WHERE bed_id = p_bed_id;
    
    -- Initialize recommendations array
    v_recommendations := ARRAY[]::TEXT[];
    
    -- Assess quality grade
    IF v_uniformity_score >= 90 AND v_coverage_percentage >= 95 AND 
       v_thickness_consistency >= 90 AND v_defect_count = 0 THEN
        quality_grade := 'EXCELLENT';
    ELSIF v_uniformity_score >= 80 AND v_coverage_percentage >= 90 AND 
          v_thickness_consistency >= 80 AND v_defect_count <= 2 THEN
        quality_grade := 'GOOD';
    ELSIF v_uniformity_score >= 70 AND v_coverage_percentage >= 85 AND 
          v_thickness_consistency >= 70 AND v_defect_count <= 5 THEN
        quality_grade := 'ACCEPTABLE';
    ELSE
        quality_grade := 'POOR';
    END IF;
    
    -- Generate recommendations
    IF v_uniformity_score < 80 THEN
        v_recommendations := array_append(v_recommendations, 'Improve powder distribution uniformity');
    END IF;
    
    IF v_coverage_percentage < 90 THEN
        v_recommendations := array_append(v_recommendations, 'Increase powder coverage percentage');
    END IF;
    
    IF v_thickness_consistency < 80 THEN
        v_recommendations := array_append(v_recommendations, 'Improve layer thickness consistency');
    END IF;
    
    IF v_defect_count > 5 THEN
        v_recommendations := array_append(v_recommendations, 'Address powder bed defects');
    END IF;
    
    IF v_surface_roughness > 20 THEN
        v_recommendations := array_append(v_recommendations, 'Reduce surface roughness');
    END IF;
    
    RETURN QUERY SELECT quality_grade, v_recommendations;
END;
$$ LANGUAGE plpgsql;

-- Create comments for documentation
COMMENT ON TABLE powder_bed_data IS 'Powder bed monitoring data for PBF-LB/M additive manufacturing';
COMMENT ON COLUMN powder_bed_data.bed_id IS 'Unique identifier for the powder bed record';
COMMENT ON COLUMN powder_bed_data.process_id IS 'Associated PBF process identifier';
COMMENT ON COLUMN powder_bed_data.layer_number IS 'Current layer number';
COMMENT ON COLUMN powder_bed_data.timestamp IS 'Monitoring timestamp';
COMMENT ON COLUMN powder_bed_data.image_id IS 'Unique identifier for the powder bed image';
COMMENT ON COLUMN powder_bed_data.camera_id IS 'Camera identifier used for capture';
COMMENT ON COLUMN powder_bed_data.image_format IS 'Image file format';
COMMENT ON COLUMN powder_bed_data.resolution IS 'Image resolution';
COMMENT ON COLUMN powder_bed_data.file_size IS 'Image file size in bytes';
COMMENT ON COLUMN powder_bed_data.file_path IS 'Path to the image file';
COMMENT ON COLUMN powder_bed_data.material_type IS 'Powder material type';
COMMENT ON COLUMN powder_bed_data.uniformity_score IS 'Powder bed uniformity score (0-100)';
COMMENT ON COLUMN powder_bed_data.coverage_percentage IS 'Powder coverage percentage';
COMMENT ON COLUMN powder_bed_data.thickness_consistency IS 'Layer thickness consistency score (0-100)';
COMMENT ON COLUMN powder_bed_data.surface_roughness IS 'Surface roughness in micrometers';
COMMENT ON COLUMN powder_bed_data.defects_detected IS 'Whether defects were detected';
COMMENT ON COLUMN powder_bed_data.defect_count IS 'Number of detected defects';
COMMENT ON COLUMN powder_bed_data.overall_quality_assessment IS 'Overall powder bed quality assessment';
COMMENT ON COLUMN powder_bed_data.processing_status IS 'Powder bed processing status';
COMMENT ON COLUMN powder_bed_data.metadata IS 'Additional metadata as JSON';
