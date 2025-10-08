-- Fact table for process analytics
-- This model provides comprehensive process analytics for business intelligence and reporting

{{ config(
    materialized='table',
    tags=['marts', 'fact', 'process_analytics']
) }}

with process_parameters as (
    select * from {{ ref('int_process_parameters') }}
),

quality_metrics as (
    select * from {{ ref('int_quality_metrics') }}
),

defect_analysis as (
    select * from {{ ref('int_defect_analysis') }}
),

-- Combine all data sources
combined_data as (
    select
        coalesce(p.process_id, q.process_id, d.process_id) as process_id,
        
        -- Process parameters
        p.machine_id,
        p.build_id,
        p.timestamp,
        p.layer_number,
        p.temperature,
        p.pressure,
        p.laser_power,
        p.scan_speed,
        p.layer_height,
        p.hatch_spacing,
        p.exposure_time,
        p.atmosphere,
        p.powder_material,
        p.powder_batch_id,
        p.process_efficiency,
        p.estimated_energy_consumption_kwh,
        p.quality_grade,
        p.data_completeness_score,
        
        -- ISPM sensor data
        p.avg_thermal_measurement,
        p.thermal_measurement_stddev,
        p.thermal_signal_quality,
        p.thermal_risk_score,
        p.avg_optical_measurement,
        p.optical_measurement_stddev,
        p.optical_signal_quality,
        p.optical_risk_score,
        p.avg_acoustic_measurement,
        p.acoustic_measurement_stddev,
        p.acoustic_signal_quality,
        p.acoustic_risk_score,
        p.avg_vibration_measurement,
        p.vibration_measurement_stddev,
        p.vibration_signal_quality,
        p.vibration_risk_score,
        p.avg_pressure_measurement,
        p.pressure_measurement_stddev,
        p.pressure_signal_quality,
        p.pressure_risk_score,
        p.avg_melt_pool_measurement,
        p.melt_pool_measurement_stddev,
        p.melt_pool_signal_quality,
        p.melt_pool_risk_score,
        p.avg_layer_height_measurement,
        p.layer_height_measurement_stddev,
        p.layer_height_signal_quality,
        p.layer_height_risk_score,
        p.overall_signal_quality,
        p.overall_measurement_confidence,
        p.overall_max_risk_score,
        p.total_measurements,
        p.thermal_stability,
        p.pressure_stability,
        p.laser_optical_alignment,
        p.parameter_optimization_score,
        
        -- Quality metrics
        q.total_layers,
        q.first_timestamp,
        q.last_timestamp,
        q.min_layer,
        q.max_layer,
        q.avg_density,
        q.min_density,
        q.max_density,
        q.density_stddev,
        q.avg_surface_roughness,
        q.min_surface_roughness,
        q.max_surface_roughness,
        q.surface_roughness_stddev,
        q.avg_dimensional_accuracy,
        q.min_dimensional_accuracy,
        q.max_dimensional_accuracy,
        q.dimensional_accuracy_stddev,
        q.pbf_total_defects,
        q.pbf_avg_defect_count,
        q.pbf_max_defect_count,
        q.layers_with_defects,
        q.excellent_layers,
        q.good_layers,
        q.acceptable_layers,
        q.poor_layers,
        q.pbf_avg_data_completeness_score,
        q.total_scans,
        q.first_scan,
        q.last_scan,
        q.avg_ct_quality_score,
        q.min_ct_quality_score,
        q.max_ct_quality_score,
        q.ct_quality_score_stddev,
        q.total_ct_defects,
        q.avg_ct_defects,
        q.max_ct_defects,
        q.accepted_scans,
        q.rejected_scans,
        q.conditional_scans,
        q.review_required_scans,
        q.avg_contrast_to_noise_ratio,
        q.avg_signal_to_noise_ratio,
        q.avg_spatial_resolution,
        q.avg_uniformity,
        q.scans_with_artifacts,
        q.avg_artifact_penalty,
        q.excellent_scans,
        q.good_scans,
        q.acceptable_scans,
        q.poor_scans,
        q.avg_ct_data_completeness_score,
        q.total_powder_bed_records,
        q.first_powder_bed_timestamp,
        q.last_powder_bed_timestamp,
        q.min_pb_layer,
        q.max_pb_layer,
        q.avg_uniformity_score,
        q.min_uniformity_score,
        q.max_uniformity_score,
        q.uniformity_stddev,
        q.avg_coverage_percentage,
        q.min_coverage_percentage,
        q.max_coverage_percentage,
        q.coverage_stddev,
        q.avg_thickness_consistency,
        q.min_thickness_consistency,
        q.max_thickness_consistency,
        q.thickness_consistency_stddev,
        q.avg_pb_surface_roughness,
        q.min_pb_surface_roughness,
        q.max_pb_surface_roughness,
        q.pb_surface_roughness_stddev,
        q.total_pb_defects,
        q.avg_pb_defect_count,
        q.max_pb_defects,
        q.powder_bed_records_with_defects,
        q.excellent_pb_records,
        q.good_pb_records,
        q.acceptable_pb_records,
        q.poor_pb_records,
        q.unacceptable_pb_records,
        q.avg_image_quality_score,
        q.min_image_quality_score,
        q.max_image_quality_score,
        q.avg_powder_flowability_score,
        q.avg_moisture_risk_score,
        q.avg_environmental_risk_score,
        q.avg_pb_data_completeness_score,
        q.overall_quality_score,
        q.quality_consistency_score,
        q.defect_rate_percentage,
        q.acceptance_rate_percentage,
        q.overall_data_completeness_score,
        
        -- Defect analysis
        d.total_defect_records,
        d.first_defect_timestamp,
        d.last_defect_timestamp,
        d.first_defect_layer,
        d.last_defect_layer,
        d.total_pbf_defects as defect_total_pbf_defects,
        d.avg_defects_per_layer,
        d.max_defects_per_layer,
        d.min_defects_per_layer,
        d.defect_count_stddev,
        d.avg_density_at_defects,
        d.min_density_at_defects,
        d.max_density_at_defects,
        d.avg_surface_roughness_at_defects,
        d.min_surface_roughness_at_defects,
        d.max_surface_roughness_at_defects,
        d.avg_dimensional_accuracy_at_defects,
        d.min_dimensional_accuracy_at_defects,
        d.max_dimensional_accuracy_at_defects,
        d.excellent_layers_with_defects,
        d.good_layers_with_defects,
        d.acceptable_layers_with_defects,
        d.poor_layers_with_defects,
        d.single_defect_layers,
        d.moderate_defect_layers,
        d.high_defect_layers,
        d.critical_defect_layers,
        d.total_ct_defect_records,
        d.first_ct_defect_timestamp,
        d.last_ct_defect_timestamp,
        d.total_ct_defects as defect_total_ct_defects,
        d.avg_ct_defects_per_scan,
        d.max_ct_defects_per_scan,
        d.min_ct_defects_per_scan,
        d.ct_defect_count_stddev,
        d.avg_quality_score_at_defects,
        d.min_quality_score_at_defects,
        d.max_quality_score_at_defects,
        d.accepted_scans_with_defects,
        d.rejected_scans_with_defects,
        d.conditional_scans_with_defects,
        d.review_required_scans_with_defects,
        d.scans_with_artifacts,
        d.no_artifact_scans,
        d.minimal_artifact_scans,
        d.moderate_artifact_scans,
        d.severe_artifact_scans,
        d.excellent_scans_with_defects,
        d.good_scans_with_defects,
        d.acceptable_scans_with_defects,
        d.poor_scans_with_defects,
        d.single_defect_scans,
        d.moderate_defect_scans,
        d.high_defect_scans,
        d.critical_defect_scans,
        d.total_pb_defect_records,
        d.first_pb_defect_timestamp,
        d.last_pb_defect_timestamp,
        d.first_pb_defect_layer,
        d.last_pb_defect_layer,
        d.total_pb_defects as defect_total_pb_defects,
        d.avg_pb_defects_per_record,
        d.max_pb_defects_per_record,
        d.min_pb_defects_per_record,
        d.pb_defect_count_stddev,
        d.avg_uniformity_at_defects,
        d.min_uniformity_at_defects,
        d.max_uniformity_at_defects,
        d.avg_coverage_at_defects,
        d.min_coverage_at_defects,
        d.max_coverage_at_defects,
        d.avg_thickness_consistency_at_defects,
        d.min_thickness_consistency_at_defects,
        d.max_thickness_consistency_at_defects,
        d.avg_pb_surface_roughness_at_defects,
        d.min_pb_surface_roughness_at_defects,
        d.max_pb_surface_roughness_at_defects,
        d.avg_density_variation_at_defects,
        d.avg_defect_density,
        d.avg_moisture_risk_at_defects,
        d.avg_environmental_risk_at_defects,
        d.excellent_pb_records_with_defects,
        d.good_pb_records_with_defects,
        d.acceptable_pb_records_with_defects,
        d.poor_pb_records_with_defects,
        d.unacceptable_pb_records_with_defects,
        d.single_pb_defect_records,
        d.moderate_pb_defect_records,
        d.high_pb_defect_records,
        d.critical_pb_defect_records,
        d.overall_defect_severity_score,
        d.total_defects_across_all_sources,
        d.density_uniformity_correlation,
        d.overall_risk_assessment,
        d.defect_pattern_type
        
    from process_parameters p
    full outer join quality_metrics q on p.process_id = q.process_id
    full outer join defect_analysis d on coalesce(p.process_id, q.process_id) = d.process_id
)

select
    *,
    
    -- Process performance score
    (
        coalesce(process_efficiency, 0) * 0.2 +
        coalesce(parameter_optimization_score, 0) * 0.2 +
        coalesce(overall_quality_score, 0) * 0.3 +
        coalesce(quality_consistency_score, 0) * 0.2 +
        coalesce(overall_data_completeness_score, 0) * 0.1
    ) as process_performance_score,
    
    -- Process efficiency rating
    case 
        when process_efficiency >= 0.9 then 'EXCELLENT'
        when process_efficiency >= 0.8 then 'GOOD'
        when process_efficiency >= 0.7 then 'ACCEPTABLE'
        when process_efficiency >= 0.6 then 'POOR'
        else 'UNACCEPTABLE'
    end as process_efficiency_rating,
    
    -- Quality rating
    case 
        when overall_quality_score >= 90 then 'EXCELLENT'
        when overall_quality_score >= 80 then 'GOOD'
        when overall_quality_score >= 70 then 'ACCEPTABLE'
        when overall_quality_score >= 60 then 'POOR'
        else 'UNACCEPTABLE'
    end as quality_rating,
    
    -- Defect risk rating
    case 
        when overall_defect_severity_score is null or overall_defect_severity_score = 0 then 'NO_RISK'
        when overall_defect_severity_score <= 10 then 'LOW_RISK'
        when overall_defect_severity_score <= 30 then 'MODERATE_RISK'
        when overall_defect_severity_score <= 60 then 'HIGH_RISK'
        else 'CRITICAL_RISK'
    end as defect_risk_rating,
    
    -- Process stability rating
    case 
        when quality_consistency_score >= 90 then 'HIGHLY_STABLE'
        when quality_consistency_score >= 80 then 'STABLE'
        when quality_consistency_score >= 70 then 'MODERATELY_STABLE'
        when quality_consistency_score >= 60 then 'UNSTABLE'
        else 'HIGHLY_UNSTABLE'
    end as process_stability_rating,
    
    -- Overall process rating
    case 
        when process_performance_score >= 90 then 'EXCELLENT'
        when process_performance_score >= 80 then 'GOOD'
        when process_performance_score >= 70 then 'ACCEPTABLE'
        when process_performance_score >= 60 then 'POOR'
        else 'UNACCEPTABLE'
    end as overall_process_rating,
    
    -- Process duration in hours
    case 
        when first_timestamp is not null and last_timestamp is not null then
            extract(epoch from (last_timestamp - first_timestamp)) / 3600.0
        else null
    end as process_duration_hours,
    
    -- Layer processing rate
    case 
        when total_layers is not null and process_duration_hours is not null and process_duration_hours > 0 then
            total_layers / process_duration_hours
        else null
    end as layer_processing_rate,
    
    -- Energy efficiency
    case 
        when estimated_energy_consumption_kwh is not null and total_layers is not null and total_layers > 0 then
            estimated_energy_consumption_kwh / total_layers
        else null
    end as energy_per_layer_kwh,
    
    -- Cost per layer (estimated)
    case 
        when energy_per_layer_kwh is not null then
            energy_per_layer_kwh * 0.12  -- Assuming $0.12 per kWh
        else null
    end as estimated_cost_per_layer_usd,
    
    -- Quality yield percentage
    case 
        when total_layers is not null and total_layers > 0 then
            ((excellent_layers + good_layers + acceptable_layers)::float / total_layers) * 100
        else null
    end as quality_yield_percentage,
    
    -- Defect rate percentage
    case 
        when total_layers is not null and total_layers > 0 then
            (layers_with_defects::float / total_layers) * 100
        else null
    end as defect_rate_percentage,
    
    -- Acceptance rate percentage
    case 
        when total_scans is not null and total_scans > 0 then
            (accepted_scans::float / total_scans) * 100
        else null
    end as acceptance_rate_percentage,
    
    -- Process optimization recommendations
    case 
        when process_efficiency < 0.7 then 'OPTIMIZE_PROCESS_PARAMETERS'
        when overall_quality_score < 70 then 'IMPROVE_QUALITY_CONTROL'
        when defect_rate_percentage > 10 then 'ADDRESS_DEFECT_ISSUES'
        when quality_consistency_score < 70 then 'IMPROVE_PROCESS_STABILITY'
        when overall_max_risk_score > 50 then 'REDUCE_RISK_FACTORS'
        else 'MAINTAIN_CURRENT_SETTINGS'
    end as optimization_recommendation,
    
    -- Process status
    case 
        when overall_process_rating = 'EXCELLENT' then 'OPTIMAL'
        when overall_process_rating = 'GOOD' then 'SATISFACTORY'
        when overall_process_rating = 'ACCEPTABLE' then 'NEEDS_IMPROVEMENT'
        when overall_process_rating = 'POOR' then 'REQUIRES_ATTENTION'
        else 'CRITICAL_ISSUES'
    end as process_status

from combined_data
