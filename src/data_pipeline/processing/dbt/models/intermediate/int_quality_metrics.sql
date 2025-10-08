-- Intermediate model for quality metrics
-- This model aggregates and analyzes quality metrics across different data sources

{{ config(
    materialized='table',
    tags=['intermediate', 'quality_metrics']
) }}

with pbf_process_quality as (
    select
        process_id,
        machine_id,
        build_id,
        timestamp,
        layer_number,
        density,
        surface_roughness,
        dimensional_accuracy,
        defect_count,
        quality_grade,
        data_completeness_score
    from {{ ref('stg_pbf_process_data') }}
    where density is not null or surface_roughness is not null or 
          dimensional_accuracy is not null or defect_count is not null
),

ct_scan_quality as (
    select
        scan_id,
        process_id,
        part_id,
        created_at,
        overall_quality_score,
        total_defects,
        acceptance_status,
        dimensional_accuracy as ct_dimensional_accuracy,
        contrast_to_noise_ratio,
        signal_to_noise_ratio,
        spatial_resolution,
        uniformity,
        artifacts_detected,
        artifact_severity,
        calculated_quality_score,
        artifact_penalty,
        quality_grade as ct_quality_grade,
        data_completeness_score as ct_data_completeness_score
    from {{ ref('stg_ct_scan_data') }}
),

powder_bed_quality as (
    select
        bed_id,
        process_id,
        layer_number,
        timestamp,
        material_type,
        uniformity_score,
        coverage_percentage,
        thickness_consistency,
        surface_roughness as pb_surface_roughness,
        density_variation,
        defect_density,
        defects_detected,
        defect_count as pb_defect_count,
        overall_quality_assessment,
        calculated_quality_score as pb_calculated_quality_score,
        image_quality_score,
        powder_flowability_score,
        moisture_risk_score,
        environmental_risk_score,
        data_completeness_score as pb_data_completeness_score
    from {{ ref('stg_powder_bed_data') }}
),

-- Aggregate quality metrics by process
pbf_quality_aggregated as (
    select
        process_id,
        machine_id,
        build_id,
        count(*) as total_layers,
        min(timestamp) as first_timestamp,
        max(timestamp) as last_timestamp,
        min(layer_number) as min_layer,
        max(layer_number) as max_layer,
        
        -- Density metrics
        avg(density) as avg_density,
        min(density) as min_density,
        max(density) as max_density,
        stddev(density) as density_stddev,
        
        -- Surface roughness metrics
        avg(surface_roughness) as avg_surface_roughness,
        min(surface_roughness) as min_surface_roughness,
        max(surface_roughness) as max_surface_roughness,
        stddev(surface_roughness) as surface_roughness_stddev,
        
        -- Dimensional accuracy metrics
        avg(dimensional_accuracy) as avg_dimensional_accuracy,
        min(dimensional_accuracy) as min_dimensional_accuracy,
        max(dimensional_accuracy) as max_dimensional_accuracy,
        stddev(dimensional_accuracy) as dimensional_accuracy_stddev,
        
        -- Defect metrics
        sum(defect_count) as total_defects,
        avg(defect_count) as avg_defect_count,
        max(defect_count) as max_defect_count,
        count(case when defect_count > 0 then 1 end) as layers_with_defects,
        
        -- Quality grade distribution
        count(case when quality_grade = 'EXCELLENT' then 1 end) as excellent_layers,
        count(case when quality_grade = 'GOOD' then 1 end) as good_layers,
        count(case when quality_grade = 'ACCEPTABLE' then 1 end) as acceptable_layers,
        count(case when quality_grade = 'POOR' then 1 end) as poor_layers,
        
        -- Data completeness
        avg(data_completeness_score) as avg_data_completeness_score
        
    from pbf_process_quality
    group by process_id, machine_id, build_id
),

-- Aggregate CT scan quality by process
ct_quality_aggregated as (
    select
        process_id,
        count(*) as total_scans,
        min(created_at) as first_scan,
        max(created_at) as last_scan,
        
        -- Quality score metrics
        avg(overall_quality_score) as avg_ct_quality_score,
        min(overall_quality_score) as min_ct_quality_score,
        max(overall_quality_score) as max_ct_quality_score,
        stddev(overall_quality_score) as ct_quality_score_stddev,
        
        -- Defect metrics
        sum(total_defects) as total_ct_defects,
        avg(total_defects) as avg_ct_defects,
        max(total_defects) as max_ct_defects,
        
        -- Acceptance status distribution
        count(case when acceptance_status = 'ACCEPTED' then 1 end) as accepted_scans,
        count(case when acceptance_status = 'REJECTED' then 1 end) as rejected_scans,
        count(case when acceptance_status = 'CONDITIONAL' then 1 end) as conditional_scans,
        count(case when acceptance_status = 'REQUIRES_REVIEW' then 1 end) as review_required_scans,
        
        -- Image quality metrics
        avg(contrast_to_noise_ratio) as avg_contrast_to_noise_ratio,
        avg(signal_to_noise_ratio) as avg_signal_to_noise_ratio,
        avg(spatial_resolution) as avg_spatial_resolution,
        avg(uniformity) as avg_uniformity,
        
        -- Artifact metrics
        count(case when artifacts_detected = true then 1 end) as scans_with_artifacts,
        avg(artifact_penalty) as avg_artifact_penalty,
        
        -- Quality grade distribution
        count(case when ct_quality_grade = 'EXCELLENT' then 1 end) as excellent_scans,
        count(case when ct_quality_grade = 'GOOD' then 1 end) as good_scans,
        count(case when ct_quality_grade = 'ACCEPTABLE' then 1 end) as acceptable_scans,
        count(case when ct_quality_grade = 'POOR' then 1 end) as poor_scans,
        
        -- Data completeness
        avg(ct_data_completeness_score) as avg_ct_data_completeness_score
        
    from ct_scan_quality
    group by process_id
),

-- Aggregate powder bed quality by process
powder_bed_quality_aggregated as (
    select
        process_id,
        count(*) as total_powder_bed_records,
        min(timestamp) as first_powder_bed_timestamp,
        max(timestamp) as last_powder_bed_timestamp,
        min(layer_number) as min_pb_layer,
        max(layer_number) as max_pb_layer,
        
        -- Uniformity metrics
        avg(uniformity_score) as avg_uniformity_score,
        min(uniformity_score) as min_uniformity_score,
        max(uniformity_score) as max_uniformity_score,
        stddev(uniformity_score) as uniformity_stddev,
        
        -- Coverage metrics
        avg(coverage_percentage) as avg_coverage_percentage,
        min(coverage_percentage) as min_coverage_percentage,
        max(coverage_percentage) as max_coverage_percentage,
        stddev(coverage_percentage) as coverage_stddev,
        
        -- Thickness consistency metrics
        avg(thickness_consistency) as avg_thickness_consistency,
        min(thickness_consistency) as min_thickness_consistency,
        max(thickness_consistency) as max_thickness_consistency,
        stddev(thickness_consistency) as thickness_consistency_stddev,
        
        -- Surface roughness metrics
        avg(pb_surface_roughness) as avg_pb_surface_roughness,
        min(pb_surface_roughness) as min_pb_surface_roughness,
        max(pb_surface_roughness) as max_pb_surface_roughness,
        stddev(pb_surface_roughness) as pb_surface_roughness_stddev,
        
        -- Defect metrics
        sum(pb_defect_count) as total_pb_defects,
        avg(pb_defect_count) as avg_pb_defect_count,
        max(pb_defect_count) as max_pb_defects,
        count(case when defects_detected = true then 1 end) as powder_bed_records_with_defects,
        
        -- Quality assessment distribution
        count(case when overall_quality_assessment = 'EXCELLENT' then 1 end) as excellent_pb_records,
        count(case when overall_quality_assessment = 'GOOD' then 1 end) as good_pb_records,
        count(case when overall_quality_assessment = 'ACCEPTABLE' then 1 end) as acceptable_pb_records,
        count(case when overall_quality_assessment = 'POOR' then 1 end) as poor_pb_records,
        count(case when overall_quality_assessment = 'UNACCEPTABLE' then 1 end) as unacceptable_pb_records,
        
        -- Image quality metrics
        avg(image_quality_score) as avg_image_quality_score,
        min(image_quality_score) as min_image_quality_score,
        max(image_quality_score) as max_image_quality_score,
        
        -- Powder characteristics
        avg(powder_flowability_score) as avg_powder_flowability_score,
        avg(moisture_risk_score) as avg_moisture_risk_score,
        avg(environmental_risk_score) as avg_environmental_risk_score,
        
        -- Data completeness
        avg(pb_data_completeness_score) as avg_pb_data_completeness_score
        
    from powder_bed_quality
    group by process_id
),

-- Combine all quality metrics
combined_quality_metrics as (
    select
        coalesce(p.process_id, c.process_id, pb.process_id) as process_id,
        
        -- PBF process quality metrics
        p.machine_id,
        p.build_id,
        p.total_layers,
        p.first_timestamp,
        p.last_timestamp,
        p.min_layer,
        p.max_layer,
        
        p.avg_density,
        p.min_density,
        p.max_density,
        p.density_stddev,
        
        p.avg_surface_roughness,
        p.min_surface_roughness,
        p.max_surface_roughness,
        p.surface_roughness_stddev,
        
        p.avg_dimensional_accuracy,
        p.min_dimensional_accuracy,
        p.max_dimensional_accuracy,
        p.dimensional_accuracy_stddev,
        
        p.total_defects as pbf_total_defects,
        p.avg_defect_count as pbf_avg_defect_count,
        p.max_defect_count as pbf_max_defect_count,
        p.layers_with_defects,
        
        p.excellent_layers,
        p.good_layers,
        p.acceptable_layers,
        p.poor_layers,
        
        p.avg_data_completeness_score as pbf_avg_data_completeness_score,
        
        -- CT scan quality metrics
        c.total_scans,
        c.first_scan,
        c.last_scan,
        
        c.avg_ct_quality_score,
        c.min_ct_quality_score,
        c.max_ct_quality_score,
        c.ct_quality_score_stddev,
        
        c.total_ct_defects,
        c.avg_ct_defects,
        c.max_ct_defects,
        
        c.accepted_scans,
        c.rejected_scans,
        c.conditional_scans,
        c.review_required_scans,
        
        c.avg_contrast_to_noise_ratio,
        c.avg_signal_to_noise_ratio,
        c.avg_spatial_resolution,
        c.avg_uniformity,
        
        c.scans_with_artifacts,
        c.avg_artifact_penalty,
        
        c.excellent_scans,
        c.good_scans,
        c.acceptable_scans,
        c.poor_scans,
        
        c.avg_ct_data_completeness_score,
        
        -- Powder bed quality metrics
        pb.total_powder_bed_records,
        pb.first_powder_bed_timestamp,
        pb.last_powder_bed_timestamp,
        pb.min_pb_layer,
        pb.max_pb_layer,
        
        pb.avg_uniformity_score,
        pb.min_uniformity_score,
        pb.max_uniformity_score,
        pb.uniformity_stddev,
        
        pb.avg_coverage_percentage,
        pb.min_coverage_percentage,
        pb.max_coverage_percentage,
        pb.coverage_stddev,
        
        pb.avg_thickness_consistency,
        pb.min_thickness_consistency,
        pb.max_thickness_consistency,
        pb.thickness_consistency_stddev,
        
        pb.avg_pb_surface_roughness,
        pb.min_pb_surface_roughness,
        pb.max_pb_surface_roughness,
        pb.pb_surface_roughness_stddev,
        
        pb.total_pb_defects,
        pb.avg_pb_defect_count,
        pb.max_pb_defects,
        pb.powder_bed_records_with_defects,
        
        pb.excellent_pb_records,
        pb.good_pb_records,
        pb.acceptable_pb_records,
        pb.poor_pb_records,
        pb.unacceptable_pb_records,
        
        pb.avg_image_quality_score,
        pb.min_image_quality_score,
        pb.max_image_quality_score,
        
        pb.avg_powder_flowability_score,
        pb.avg_moisture_risk_score,
        pb.avg_environmental_risk_score,
        
        pb.avg_pb_data_completeness_score
        
    from pbf_quality_aggregated p
    full outer join ct_quality_aggregated c on p.process_id = c.process_id
    full outer join powder_bed_quality_aggregated pb on coalesce(p.process_id, c.process_id) = pb.process_id
)

select
    *,
    
    -- Overall quality score calculation
    (
        coalesce(avg_density, 0) * 0.25 +
        coalesce(100 - avg_surface_roughness, 0) * 0.15 +
        coalesce(100 - (avg_dimensional_accuracy * 100), 0) * 0.15 +
        coalesce(avg_ct_quality_score, 0) * 0.25 +
        coalesce(avg_uniformity_score, 0) * 0.1 +
        coalesce(avg_coverage_percentage, 0) * 0.1
    ) as overall_quality_score,
    
    -- Quality consistency score
    (
        case when density_stddev is not null then greatest(0, 100 - density_stddev * 10) else 0 end +
        case when surface_roughness_stddev is not null then greatest(0, 100 - surface_roughness_stddev * 5) else 0 end +
        case when dimensional_accuracy_stddev is not null then greatest(0, 100 - dimensional_accuracy_stddev * 100) else 0 end +
        case when ct_quality_score_stddev is not null then greatest(0, 100 - ct_quality_score_stddev * 2) else 0 end +
        case when uniformity_stddev is not null then greatest(0, 100 - uniformity_stddev * 2) else 0 end +
        case when coverage_stddev is not null then greatest(0, 100 - coverage_stddev * 2) else 0 end
    ) / 6.0 as quality_consistency_score,
    
    -- Defect rate calculation
    case 
        when total_layers > 0 then (layers_with_defects::float / total_layers) * 100
        else null
    end as defect_rate_percentage,
    
    -- Acceptance rate calculation
    case 
        when total_scans > 0 then (accepted_scans::float / total_scans) * 100
        else null
    end as acceptance_rate_percentage,
    
    -- Overall data completeness score
    (
        coalesce(pbf_avg_data_completeness_score, 0) * 0.4 +
        coalesce(avg_ct_data_completeness_score, 0) * 0.3 +
        coalesce(avg_pb_data_completeness_score, 0) * 0.3
    ) as overall_data_completeness_score

from combined_quality_metrics
