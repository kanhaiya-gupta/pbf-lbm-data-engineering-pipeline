-- Dimension table for quality standards
-- This model provides standardized quality standards and thresholds for dimensional modeling

{{ config(
    materialized='table',
    tags=['marts', 'dimension', 'quality_standards']
) }}

with quality_metrics as (
    select * from {{ ref('int_quality_metrics') }}
),

-- Create dimension table with standardized quality standards and thresholds
dimension_data as (
    select
        process_id,
        
        -- Quality metrics
        avg_density,
        min_density,
        max_density,
        density_stddev,
        avg_surface_roughness,
        min_surface_roughness,
        max_surface_roughness,
        surface_roughness_stddev,
        avg_dimensional_accuracy,
        min_dimensional_accuracy,
        max_dimensional_accuracy,
        dimensional_accuracy_stddev,
        pbf_total_defects,
        pbf_avg_defect_count,
        pbf_max_defect_count,
        layers_with_defects,
        excellent_layers,
        good_layers,
        acceptable_layers,
        poor_layers,
        avg_ct_quality_score,
        min_ct_quality_score,
        max_ct_quality_score,
        ct_quality_score_stddev,
        total_ct_defects,
        avg_ct_defects,
        max_ct_defects,
        accepted_scans,
        rejected_scans,
        conditional_scans,
        review_required_scans,
        avg_contrast_to_noise_ratio,
        avg_signal_to_noise_ratio,
        avg_spatial_resolution,
        avg_uniformity,
        scans_with_artifacts,
        avg_artifact_penalty,
        excellent_scans,
        good_scans,
        acceptable_scans,
        poor_scans,
        avg_uniformity_score,
        min_uniformity_score,
        max_uniformity_score,
        uniformity_stddev,
        avg_coverage_percentage,
        min_coverage_percentage,
        max_coverage_percentage,
        coverage_stddev,
        avg_thickness_consistency,
        min_thickness_consistency,
        max_thickness_consistency,
        thickness_consistency_stddev,
        avg_pb_surface_roughness,
        min_pb_surface_roughness,
        max_pb_surface_roughness,
        pb_surface_roughness_stddev,
        total_pb_defects,
        avg_pb_defect_count,
        max_pb_defects,
        powder_bed_records_with_defects,
        excellent_pb_records,
        good_pb_records,
        acceptable_pb_records,
        poor_pb_records,
        unacceptable_pb_records,
        avg_image_quality_score,
        min_image_quality_score,
        max_image_quality_score,
        avg_powder_flowability_score,
        avg_moisture_risk_score,
        avg_environmental_risk_score,
        overall_quality_score,
        quality_consistency_score,
        defect_rate_percentage,
        acceptance_rate_percentage,
        overall_data_completeness_score,
        
        -- Quality grade categories
        case 
            when avg_density >= 95 then 'EXCELLENT_DENSITY'
            when avg_density >= 90 then 'GOOD_DENSITY'
            when avg_density >= 85 then 'ACCEPTABLE_DENSITY'
            when avg_density >= 80 then 'POOR_DENSITY'
            else 'UNACCEPTABLE_DENSITY'
        end as density_grade,
        
        case 
            when avg_surface_roughness <= 5 then 'EXCELLENT_SURFACE'
            when avg_surface_roughness <= 10 then 'GOOD_SURFACE'
            when avg_surface_roughness <= 20 then 'ACCEPTABLE_SURFACE'
            when avg_surface_roughness <= 30 then 'POOR_SURFACE'
            else 'UNACCEPTABLE_SURFACE'
        end as surface_roughness_grade,
        
        case 
            when avg_dimensional_accuracy <= 0.05 then 'EXCELLENT_DIMENSIONAL'
            when avg_dimensional_accuracy <= 0.1 then 'GOOD_DIMENSIONAL'
            when avg_dimensional_accuracy <= 0.2 then 'ACCEPTABLE_DIMENSIONAL'
            when avg_dimensional_accuracy <= 0.5 then 'POOR_DIMENSIONAL'
            else 'UNACCEPTABLE_DIMENSIONAL'
        end as dimensional_accuracy_grade,
        
        case 
            when pbf_avg_defect_count = 0 then 'NO_DEFECTS'
            when pbf_avg_defect_count <= 2 then 'LOW_DEFECTS'
            when pbf_avg_defect_count <= 5 then 'MODERATE_DEFECTS'
            when pbf_avg_defect_count <= 10 then 'HIGH_DEFECTS'
            else 'CRITICAL_DEFECTS'
        end as defect_level_grade,
        
        case 
            when avg_ct_quality_score >= 90 then 'EXCELLENT_CT_QUALITY'
            when avg_ct_quality_score >= 80 then 'GOOD_CT_QUALITY'
            when avg_ct_quality_score >= 70 then 'ACCEPTABLE_CT_QUALITY'
            when avg_ct_quality_score >= 60 then 'POOR_CT_QUALITY'
            else 'UNACCEPTABLE_CT_QUALITY'
        end as ct_quality_grade,
        
        case 
            when avg_uniformity_score >= 90 then 'EXCELLENT_UNIFORMITY'
            when avg_uniformity_score >= 80 then 'GOOD_UNIFORMITY'
            when avg_uniformity_score >= 70 then 'ACCEPTABLE_UNIFORMITY'
            when avg_uniformity_score >= 60 then 'POOR_UNIFORMITY'
            else 'UNACCEPTABLE_UNIFORMITY'
        end as uniformity_grade,
        
        case 
            when avg_coverage_percentage >= 95 then 'EXCELLENT_COVERAGE'
            when avg_coverage_percentage >= 90 then 'GOOD_COVERAGE'
            when avg_coverage_percentage >= 85 then 'ACCEPTABLE_COVERAGE'
            when avg_coverage_percentage >= 80 then 'POOR_COVERAGE'
            else 'UNACCEPTABLE_COVERAGE'
        end as coverage_grade,
        
        case 
            when avg_thickness_consistency >= 90 then 'EXCELLENT_THICKNESS'
            when avg_thickness_consistency >= 80 then 'GOOD_THICKNESS'
            when avg_thickness_consistency >= 70 then 'ACCEPTABLE_THICKNESS'
            when avg_thickness_consistency >= 60 then 'POOR_THICKNESS'
            else 'UNACCEPTABLE_THICKNESS'
        end as thickness_consistency_grade,
        
        -- Quality consistency categories
        case 
            when density_stddev <= 1.0 then 'HIGHLY_CONSISTENT_DENSITY'
            when density_stddev <= 2.0 then 'CONSISTENT_DENSITY'
            when density_stddev <= 5.0 then 'MODERATELY_CONSISTENT_DENSITY'
            when density_stddev <= 10.0 then 'INCONSISTENT_DENSITY'
            else 'HIGHLY_INCONSISTENT_DENSITY'
        end as density_consistency_grade,
        
        case 
            when surface_roughness_stddev <= 1.0 then 'HIGHLY_CONSISTENT_SURFACE'
            when surface_roughness_stddev <= 2.0 then 'CONSISTENT_SURFACE'
            when surface_roughness_stddev <= 5.0 then 'MODERATELY_CONSISTENT_SURFACE'
            when surface_roughness_stddev <= 10.0 then 'INCONSISTENT_SURFACE'
            else 'HIGHLY_INCONSISTENT_SURFACE'
        end as surface_consistency_grade,
        
        case 
            when dimensional_accuracy_stddev <= 0.01 then 'HIGHLY_CONSISTENT_DIMENSIONAL'
            when dimensional_accuracy_stddev <= 0.02 then 'CONSISTENT_DIMENSIONAL'
            when dimensional_accuracy_stddev <= 0.05 then 'MODERATELY_CONSISTENT_DIMENSIONAL'
            when dimensional_accuracy_stddev <= 0.1 then 'INCONSISTENT_DIMENSIONAL'
            else 'HIGHLY_INCONSISTENT_DIMENSIONAL'
        end as dimensional_consistency_grade,
        
        -- Quality yield categories
        case 
            when (excellent_layers + good_layers + acceptable_layers)::float / total_layers >= 0.95 then 'EXCELLENT_YIELD'
            when (excellent_layers + good_layers + acceptable_layers)::float / total_layers >= 0.90 then 'GOOD_YIELD'
            when (excellent_layers + good_layers + acceptable_layers)::float / total_layers >= 0.80 then 'ACCEPTABLE_YIELD'
            when (excellent_layers + good_layers + acceptable_layers)::float / total_layers >= 0.70 then 'POOR_YIELD'
            else 'UNACCEPTABLE_YIELD'
        end as quality_yield_grade,
        
        -- Acceptance rate categories
        case 
            when acceptance_rate_percentage >= 95 then 'EXCELLENT_ACCEPTANCE'
            when acceptance_rate_percentage >= 90 then 'GOOD_ACCEPTANCE'
            when acceptance_rate_percentage >= 80 then 'ACCEPTABLE_ACCEPTANCE'
            when acceptance_rate_percentage >= 70 then 'POOR_ACCEPTANCE'
            else 'UNACCEPTABLE_ACCEPTANCE'
        end as acceptance_rate_grade,
        
        -- Defect rate categories
        case 
            when defect_rate_percentage <= 2 then 'EXCELLENT_DEFECT_RATE'
            when defect_rate_percentage <= 5 then 'GOOD_DEFECT_RATE'
            when defect_rate_percentage <= 10 then 'ACCEPTABLE_DEFECT_RATE'
            when defect_rate_percentage <= 20 then 'POOR_DEFECT_RATE'
            else 'UNACCEPTABLE_DEFECT_RATE'
        end as defect_rate_grade,
        
        -- Overall quality consistency grade
        case 
            when quality_consistency_score >= 90 then 'HIGHLY_CONSISTENT'
            when quality_consistency_score >= 80 then 'CONSISTENT'
            when quality_consistency_score >= 70 then 'MODERATELY_CONSISTENT'
            when quality_consistency_score >= 60 then 'INCONSISTENT'
            else 'HIGHLY_INCONSISTENT'
        end as overall_consistency_grade,
        
        -- Quality standard compliance
        case 
            when avg_density >= 95 and avg_surface_roughness <= 10 and 
                 avg_dimensional_accuracy <= 0.1 and pbf_avg_defect_count <= 1 then 'MEETS_PREMIUM_STANDARDS'
            when avg_density >= 90 and avg_surface_roughness <= 20 and 
                 avg_dimensional_accuracy <= 0.2 and pbf_avg_defect_count <= 5 then 'MEETS_GOOD_STANDARDS'
            when avg_density >= 85 and avg_surface_roughness <= 30 and 
                 avg_dimensional_accuracy <= 0.5 and pbf_avg_defect_count <= 10 then 'MEETS_ACCEPTABLE_STANDARDS'
            when avg_density >= 80 and avg_surface_roughness <= 50 and 
                 avg_dimensional_accuracy <= 1.0 and pbf_avg_defect_count <= 20 then 'MEETS_MINIMUM_STANDARDS'
            else 'BELOW_STANDARDS'
        end as quality_standard_compliance,
        
        -- Quality risk level
        case 
            when overall_quality_score >= 90 and quality_consistency_score >= 90 and 
                 defect_rate_percentage <= 2 and acceptance_rate_percentage >= 95 then 'LOW_QUALITY_RISK'
            when overall_quality_score >= 80 and quality_consistency_score >= 80 and 
                 defect_rate_percentage <= 5 and acceptance_rate_percentage >= 90 then 'MODERATE_QUALITY_RISK'
            when overall_quality_score >= 70 and quality_consistency_score >= 70 and 
                 defect_rate_percentage <= 10 and acceptance_rate_percentage >= 80 then 'HIGH_QUALITY_RISK'
            else 'CRITICAL_QUALITY_RISK'
        end as quality_risk_level,
        
        -- Quality improvement priority
        case 
            when overall_quality_score < 70 or quality_consistency_score < 70 or 
                 defect_rate_percentage > 20 or acceptance_rate_percentage < 70 then 'HIGH_PRIORITY'
            when overall_quality_score < 80 or quality_consistency_score < 80 or 
                 defect_rate_percentage > 10 or acceptance_rate_percentage < 80 then 'MEDIUM_PRIORITY'
            when overall_quality_score < 90 or quality_consistency_score < 90 or 
                 defect_rate_percentage > 5 or acceptance_rate_percentage < 90 then 'LOW_PRIORITY'
            else 'MAINTAIN_CURRENT'
        end as quality_improvement_priority
        
    from quality_metrics
)

select
    *,
    
    -- Quality score breakdown
    case 
        when density_grade like 'EXCELLENT%' then 25
        when density_grade like 'GOOD%' then 20
        when density_grade like 'ACCEPTABLE%' then 15
        when density_grade like 'POOR%' then 10
        else 0
    end as density_score,
    
    case 
        when surface_roughness_grade like 'EXCELLENT%' then 25
        when surface_roughness_grade like 'GOOD%' then 20
        when surface_roughness_grade like 'ACCEPTABLE%' then 15
        when surface_roughness_grade like 'POOR%' then 10
        else 0
    end as surface_roughness_score,
    
    case 
        when dimensional_accuracy_grade like 'EXCELLENT%' then 25
        when dimensional_accuracy_grade like 'GOOD%' then 20
        when dimensional_accuracy_grade like 'ACCEPTABLE%' then 15
        when dimensional_accuracy_grade like 'POOR%' then 10
        else 0
    end as dimensional_accuracy_score,
    
    case 
        when defect_level_grade = 'NO_DEFECTS' then 25
        when defect_level_grade = 'LOW_DEFECTS' then 20
        when defect_level_grade = 'MODERATE_DEFECTS' then 15
        when defect_level_grade = 'HIGH_DEFECTS' then 10
        else 0
    end as defect_score,
    
    -- Quality standard tier
    case 
        when quality_standard_compliance = 'MEETS_PREMIUM_STANDARDS' then 'PREMIUM_TIER'
        when quality_standard_compliance = 'MEETS_GOOD_STANDARDS' then 'GOOD_TIER'
        when quality_standard_compliance = 'MEETS_ACCEPTABLE_STANDARDS' then 'ACCEPTABLE_TIER'
        when quality_standard_compliance = 'MEETS_MINIMUM_STANDARDS' then 'MINIMUM_TIER'
        else 'BELOW_TIER'
    end as quality_tier,
    
    -- Quality certification level
    case 
        when quality_standard_compliance = 'MEETS_PREMIUM_STANDARDS' and 
             overall_consistency_grade = 'HIGHLY_CONSISTENT' then 'CERTIFIED_PREMIUM'
        when quality_standard_compliance = 'MEETS_GOOD_STANDARDS' and 
             overall_consistency_grade in ('HIGHLY_CONSISTENT', 'CONSISTENT') then 'CERTIFIED_GOOD'
        when quality_standard_compliance = 'MEETS_ACCEPTABLE_STANDARDS' and 
             overall_consistency_grade in ('HIGHLY_CONSISTENT', 'CONSISTENT', 'MODERATELY_CONSISTENT') then 'CERTIFIED_ACCEPTABLE'
        else 'NOT_CERTIFIED'
    end as quality_certification_level,
    
    -- Quality monitoring status
    case 
        when quality_risk_level = 'LOW_QUALITY_RISK' then 'MONITOR_ROUTINELY'
        when quality_risk_level = 'MODERATE_QUALITY_RISK' then 'MONITOR_CLOSELY'
        when quality_risk_level = 'HIGH_QUALITY_RISK' then 'MONITOR_INTENSIVELY'
        else 'MONITOR_CRITICALLY'
    end as quality_monitoring_status,
    
    -- Quality action required
    case 
        when quality_improvement_priority = 'HIGH_PRIORITY' then 'IMMEDIATE_ACTION_REQUIRED'
        when quality_improvement_priority = 'MEDIUM_PRIORITY' then 'PLANNED_ACTION_REQUIRED'
        when quality_improvement_priority = 'LOW_PRIORITY' then 'CONTINUOUS_IMPROVEMENT'
        else 'MAINTAIN_STANDARDS'
    end as quality_action_required

from dimension_data
