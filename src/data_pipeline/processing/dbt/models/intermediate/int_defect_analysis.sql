-- Intermediate model for defect analysis
-- This model analyzes defects across different data sources and provides comprehensive defect insights

{{ config(
    materialized='table',
    tags=['intermediate', 'defect_analysis']
) }}

with pbf_process_defects as (
    select
        process_id,
        machine_id,
        build_id,
        timestamp,
        layer_number,
        defect_count,
        quality_grade,
        density,
        surface_roughness,
        dimensional_accuracy
    from {{ ref('stg_pbf_process_data') }}
    where defect_count > 0
),

ct_scan_defects as (
    select
        scan_id,
        process_id,
        part_id,
        created_at,
        total_defects,
        overall_quality_score,
        acceptance_status,
        artifacts_detected,
        artifact_severity,
        quality_grade as ct_quality_grade
    from {{ ref('stg_ct_scan_data') }}
    where total_defects > 0 or artifacts_detected = true
),

powder_bed_defects as (
    select
        bed_id,
        process_id,
        layer_number,
        timestamp,
        material_type,
        defect_count as pb_defect_count,
        defects_detected,
        overall_quality_assessment,
        uniformity_score,
        coverage_percentage,
        thickness_consistency,
        surface_roughness as pb_surface_roughness,
        density_variation,
        defect_density,
        moisture_risk_score,
        environmental_risk_score
    from {{ ref('stg_powder_bed_data') }}
    where defects_detected = true or pb_defect_count > 0
),

-- Aggregate PBF process defects by process
pbf_defect_analysis as (
    select
        process_id,
        machine_id,
        build_id,
        count(*) as total_defect_records,
        min(timestamp) as first_defect_timestamp,
        max(timestamp) as last_defect_timestamp,
        min(layer_number) as first_defect_layer,
        max(layer_number) as last_defect_layer,
        
        -- Defect statistics
        sum(defect_count) as total_pbf_defects,
        avg(defect_count) as avg_defects_per_layer,
        max(defect_count) as max_defects_per_layer,
        min(defect_count) as min_defects_per_layer,
        stddev(defect_count) as defect_count_stddev,
        
        -- Quality metrics at defect locations
        avg(density) as avg_density_at_defects,
        min(density) as min_density_at_defects,
        max(density) as max_density_at_defects,
        
        avg(surface_roughness) as avg_surface_roughness_at_defects,
        min(surface_roughness) as min_surface_roughness_at_defects,
        max(surface_roughness) as max_surface_roughness_at_defects,
        
        avg(dimensional_accuracy) as avg_dimensional_accuracy_at_defects,
        min(dimensional_accuracy) as min_dimensional_accuracy_at_defects,
        max(dimensional_accuracy) as max_dimensional_accuracy_at_defects,
        
        -- Quality grade distribution at defects
        count(case when quality_grade = 'EXCELLENT' then 1 end) as excellent_layers_with_defects,
        count(case when quality_grade = 'GOOD' then 1 end) as good_layers_with_defects,
        count(case when quality_grade = 'ACCEPTABLE' then 1 end) as acceptable_layers_with_defects,
        count(case when quality_grade = 'POOR' then 1 end) as poor_layers_with_defects,
        
        -- Defect severity classification
        count(case when defect_count = 1 then 1 end) as single_defect_layers,
        count(case when defect_count between 2 and 5 then 1 end) as moderate_defect_layers,
        count(case when defect_count between 6 and 10 then 1 end) as high_defect_layers,
        count(case when defect_count > 10 then 1 end) as critical_defect_layers
        
    from pbf_process_defects
    group by process_id, machine_id, build_id
),

-- Aggregate CT scan defects by process
ct_defect_analysis as (
    select
        process_id,
        count(*) as total_ct_defect_records,
        min(created_at) as first_ct_defect_timestamp,
        max(created_at) as last_ct_defect_timestamp,
        
        -- Defect statistics
        sum(total_defects) as total_ct_defects,
        avg(total_defects) as avg_ct_defects_per_scan,
        max(total_defects) as max_ct_defects_per_scan,
        min(total_defects) as min_ct_defects_per_scan,
        stddev(total_defects) as ct_defect_count_stddev,
        
        -- Quality metrics at defects
        avg(overall_quality_score) as avg_quality_score_at_defects,
        min(overall_quality_score) as min_quality_score_at_defects,
        max(overall_quality_score) as max_quality_score_at_defects,
        
        -- Acceptance status distribution
        count(case when acceptance_status = 'ACCEPTED' then 1 end) as accepted_scans_with_defects,
        count(case when acceptance_status = 'REJECTED' then 1 end) as rejected_scans_with_defects,
        count(case when acceptance_status = 'CONDITIONAL' then 1 end) as conditional_scans_with_defects,
        count(case when acceptance_status = 'REQUIRES_REVIEW' then 1 end) as review_required_scans_with_defects,
        
        -- Artifact analysis
        count(case when artifacts_detected = true then 1 end) as scans_with_artifacts,
        count(case when artifact_severity = 'NONE' then 1 end) as no_artifact_scans,
        count(case when artifact_severity = 'MINIMAL' then 1 end) as minimal_artifact_scans,
        count(case when artifact_severity = 'MODERATE' then 1 end) as moderate_artifact_scans,
        count(case when artifact_severity = 'SEVERE' then 1 end) as severe_artifact_scans,
        
        -- Quality grade distribution at defects
        count(case when ct_quality_grade = 'EXCELLENT' then 1 end) as excellent_scans_with_defects,
        count(case when ct_quality_grade = 'GOOD' then 1 end) as good_scans_with_defects,
        count(case when ct_quality_grade = 'ACCEPTABLE' then 1 end) as acceptable_scans_with_defects,
        count(case when ct_quality_grade = 'POOR' then 1 end) as poor_scans_with_defects,
        
        -- Defect severity classification
        count(case when total_defects = 1 then 1 end) as single_defect_scans,
        count(case when total_defects between 2 and 5 then 1 end) as moderate_defect_scans,
        count(case when total_defects between 6 and 10 then 1 end) as high_defect_scans,
        count(case when total_defects > 10 then 1 end) as critical_defect_scans
        
    from ct_scan_defects
    group by process_id
),

-- Aggregate powder bed defects by process
powder_bed_defect_analysis as (
    select
        process_id,
        count(*) as total_pb_defect_records,
        min(timestamp) as first_pb_defect_timestamp,
        max(timestamp) as last_pb_defect_timestamp,
        min(layer_number) as first_pb_defect_layer,
        max(layer_number) as last_pb_defect_layer,
        
        -- Defect statistics
        sum(pb_defect_count) as total_pb_defects,
        avg(pb_defect_count) as avg_pb_defects_per_record,
        max(pb_defect_count) as max_pb_defects_per_record,
        min(pb_defect_count) as min_pb_defects_per_record,
        stddev(pb_defect_count) as pb_defect_count_stddev,
        
        -- Quality metrics at defects
        avg(uniformity_score) as avg_uniformity_at_defects,
        min(uniformity_score) as min_uniformity_at_defects,
        max(uniformity_score) as max_uniformity_at_defects,
        
        avg(coverage_percentage) as avg_coverage_at_defects,
        min(coverage_percentage) as min_coverage_at_defects,
        max(coverage_percentage) as max_coverage_at_defects,
        
        avg(thickness_consistency) as avg_thickness_consistency_at_defects,
        min(thickness_consistency) as min_thickness_consistency_at_defects,
        max(thickness_consistency) as max_thickness_consistency_at_defects,
        
        avg(pb_surface_roughness) as avg_pb_surface_roughness_at_defects,
        min(pb_surface_roughness) as min_pb_surface_roughness_at_defects,
        max(pb_surface_roughness) as max_pb_surface_roughness_at_defects,
        
        avg(density_variation) as avg_density_variation_at_defects,
        avg(defect_density) as avg_defect_density,
        
        -- Risk factors at defects
        avg(moisture_risk_score) as avg_moisture_risk_at_defects,
        avg(environmental_risk_score) as avg_environmental_risk_at_defects,
        
        -- Quality assessment distribution at defects
        count(case when overall_quality_assessment = 'EXCELLENT' then 1 end) as excellent_pb_records_with_defects,
        count(case when overall_quality_assessment = 'GOOD' then 1 end) as good_pb_records_with_defects,
        count(case when overall_quality_assessment = 'ACCEPTABLE' then 1 end) as acceptable_pb_records_with_defects,
        count(case when overall_quality_assessment = 'POOR' then 1 end) as poor_pb_records_with_defects,
        count(case when overall_quality_assessment = 'UNACCEPTABLE' then 1 end) as unacceptable_pb_records_with_defects,
        
        -- Defect severity classification
        count(case when pb_defect_count = 1 then 1 end) as single_pb_defect_records,
        count(case when pb_defect_count between 2 and 5 then 1 end) as moderate_pb_defect_records,
        count(case when pb_defect_count between 6 and 10 then 1 end) as high_pb_defect_records,
        count(case when pb_defect_count > 10 then 1 end) as critical_pb_defect_records
        
    from powder_bed_defects
    group by process_id
),

-- Combine all defect analyses
combined_defect_analysis as (
    select
        coalesce(p.process_id, c.process_id, pb.process_id) as process_id,
        
        -- PBF process defect metrics
        p.machine_id,
        p.build_id,
        p.total_defect_records,
        p.first_defect_timestamp,
        p.last_defect_timestamp,
        p.first_defect_layer,
        p.last_defect_layer,
        
        p.total_pbf_defects,
        p.avg_defects_per_layer,
        p.max_defects_per_layer,
        p.min_defects_per_layer,
        p.defect_count_stddev,
        
        p.avg_density_at_defects,
        p.min_density_at_defects,
        p.max_density_at_defects,
        
        p.avg_surface_roughness_at_defects,
        p.min_surface_roughness_at_defects,
        p.max_surface_roughness_at_defects,
        
        p.avg_dimensional_accuracy_at_defects,
        p.min_dimensional_accuracy_at_defects,
        p.max_dimensional_accuracy_at_defects,
        
        p.excellent_layers_with_defects,
        p.good_layers_with_defects,
        p.acceptable_layers_with_defects,
        p.poor_layers_with_defects,
        
        p.single_defect_layers,
        p.moderate_defect_layers,
        p.high_defect_layers,
        p.critical_defect_layers,
        
        -- CT scan defect metrics
        c.total_ct_defect_records,
        c.first_ct_defect_timestamp,
        c.last_ct_defect_timestamp,
        
        c.total_ct_defects,
        c.avg_ct_defects_per_scan,
        c.max_ct_defects_per_scan,
        c.min_ct_defects_per_scan,
        c.ct_defect_count_stddev,
        
        c.avg_quality_score_at_defects,
        c.min_quality_score_at_defects,
        c.max_quality_score_at_defects,
        
        c.accepted_scans_with_defects,
        c.rejected_scans_with_defects,
        c.conditional_scans_with_defects,
        c.review_required_scans_with_defects,
        
        c.scans_with_artifacts,
        c.no_artifact_scans,
        c.minimal_artifact_scans,
        c.moderate_artifact_scans,
        c.severe_artifact_scans,
        
        c.excellent_scans_with_defects,
        c.good_scans_with_defects,
        c.acceptable_scans_with_defects,
        c.poor_scans_with_defects,
        
        c.single_defect_scans,
        c.moderate_defect_scans,
        c.high_defect_scans,
        c.critical_defect_scans,
        
        -- Powder bed defect metrics
        pb.total_pb_defect_records,
        pb.first_pb_defect_timestamp,
        pb.last_pb_defect_timestamp,
        pb.first_pb_defect_layer,
        pb.last_pb_defect_layer,
        
        pb.total_pb_defects,
        pb.avg_pb_defects_per_record,
        pb.max_pb_defects_per_record,
        pb.min_pb_defects_per_record,
        pb.pb_defect_count_stddev,
        
        pb.avg_uniformity_at_defects,
        pb.min_uniformity_at_defects,
        pb.max_uniformity_at_defects,
        
        pb.avg_coverage_at_defects,
        pb.min_coverage_at_defects,
        pb.max_coverage_at_defects,
        
        pb.avg_thickness_consistency_at_defects,
        pb.min_thickness_consistency_at_defects,
        pb.max_thickness_consistency_at_defects,
        
        pb.avg_pb_surface_roughness_at_defects,
        pb.min_pb_surface_roughness_at_defects,
        pb.max_pb_surface_roughness_at_defects,
        
        pb.avg_density_variation_at_defects,
        pb.avg_defect_density,
        
        pb.avg_moisture_risk_at_defects,
        pb.avg_environmental_risk_at_defects,
        
        pb.excellent_pb_records_with_defects,
        pb.good_pb_records_with_defects,
        pb.acceptable_pb_records_with_defects,
        pb.poor_pb_records_with_defects,
        pb.unacceptable_pb_records_with_defects,
        
        pb.single_pb_defect_records,
        pb.moderate_pb_defect_records,
        pb.high_pb_defect_records,
        pb.critical_pb_defect_records
        
    from pbf_defect_analysis p
    full outer join ct_defect_analysis c on p.process_id = c.process_id
    full outer join powder_bed_defect_analysis pb on coalesce(p.process_id, c.process_id) = pb.process_id
)

select
    *,
    
    -- Overall defect severity score
    (
        coalesce(single_defect_layers, 0) * 1 +
        coalesce(single_defect_scans, 0) * 1 +
        coalesce(single_pb_defect_records, 0) * 1 +
        coalesce(moderate_defect_layers, 0) * 3 +
        coalesce(moderate_defect_scans, 0) * 3 +
        coalesce(moderate_pb_defect_records, 0) * 3 +
        coalesce(high_defect_layers, 0) * 7 +
        coalesce(high_defect_scans, 0) * 7 +
        coalesce(high_pb_defect_records, 0) * 7 +
        coalesce(critical_defect_layers, 0) * 15 +
        coalesce(critical_defect_scans, 0) * 15 +
        coalesce(critical_pb_defect_records, 0) * 15
    ) as overall_defect_severity_score,
    
    -- Defect frequency analysis
    case 
        when total_pbf_defects is not null and total_ct_defects is not null and total_pb_defects is not null then
            (total_pbf_defects + total_ct_defects + total_pb_defects)
        when total_pbf_defects is not null and total_ct_defects is not null then
            (total_pbf_defects + total_ct_defects)
        when total_pbf_defects is not null and total_pb_defects is not null then
            (total_pbf_defects + total_pb_defects)
        when total_ct_defects is not null and total_pb_defects is not null then
            (total_ct_defects + total_pb_defects)
        else coalesce(total_pbf_defects, total_ct_defects, total_pb_defects, 0)
    end as total_defects_across_all_sources,
    
    -- Defect correlation analysis
    case 
        when avg_density_at_defects is not null and avg_uniformity_at_defects is not null then
            case 
                when avg_density_at_defects < 90 and avg_uniformity_at_defects < 80 then 'HIGH_CORRELATION'
                when avg_density_at_defects < 95 and avg_uniformity_at_defects < 85 then 'MODERATE_CORRELATION'
                else 'LOW_CORRELATION'
            end
        else 'INSUFFICIENT_DATA'
    end as density_uniformity_correlation,
    
    -- Risk factor analysis
    case 
        when avg_moisture_risk_at_defects is not null and avg_environmental_risk_at_defects is not null then
            case 
                when avg_moisture_risk_at_defects > 50 and avg_environmental_risk_at_defects > 50 then 'HIGH_RISK'
                when avg_moisture_risk_at_defects > 30 or avg_environmental_risk_at_defects > 30 then 'MODERATE_RISK'
                else 'LOW_RISK'
            end
        else 'UNKNOWN_RISK'
    end as overall_risk_assessment,
    
    -- Defect pattern analysis
    case 
        when single_defect_layers > 0 and critical_defect_layers = 0 then 'ISOLATED_DEFECTS'
        when critical_defect_layers > 0 then 'CRITICAL_DEFECTS'
        when high_defect_layers > 0 then 'HIGH_DEFECTS'
        when moderate_defect_layers > 0 then 'MODERATE_DEFECTS'
        else 'NO_DEFECTS'
    end as defect_pattern_type

from combined_defect_analysis
