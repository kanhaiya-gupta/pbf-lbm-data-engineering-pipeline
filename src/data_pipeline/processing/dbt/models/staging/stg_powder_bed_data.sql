-- Staging model for powder bed data
-- This model cleans and standardizes raw powder bed monitoring data

{{ config(
    materialized='view',
    tags=['staging', 'powder_bed']
) }}

with source_data as (
    select * from {{ source('raw_data', 'powder_bed_data') }}
),

cleaned_data as (
    select
        -- Primary identifiers
        bed_id,
        process_id,
        layer_number,
        
        -- Timestamps
        timestamp,
        created_at,
        updated_at,
        
        -- Image metadata
        image_id,
        camera_id,
        image_format,
        resolution,
        file_size,
        file_path,
        
        -- Camera capture settings
        exposure_time,
        aperture,
        iso,
        white_balance,
        lighting_conditions,
        
        -- Powder characteristics
        material_type,
        particle_size_d10,
        particle_size_d50,
        particle_size_d90,
        particle_size_span,
        powder_density,
        flowability,
        moisture_content,
        
        -- Bed quality metrics
        uniformity_score,
        coverage_percentage,
        thickness_consistency,
        surface_roughness,
        density_variation,
        defect_density,
        
        -- Image analysis
        brightness,
        contrast,
        sharpness,
        noise_level,
        red_channel,
        green_channel,
        blue_channel,
        
        -- Texture analysis
        texture_homogeneity,
        texture_contrast,
        texture_energy,
        texture_entropy,
        
        -- Defect detection
        defects_detected,
        defect_count,
        overall_quality_assessment,
        
        -- Environmental conditions
        ambient_temperature,
        relative_humidity,
        atmospheric_pressure,
        vibration_level,
        
        -- Processing status
        processing_status,
        metadata,
        
        -- Data quality flags
        case 
            when bed_id is null then true
            else false
        end as missing_bed_id,
        
        case 
            when process_id is null then true
            else false
        end as missing_process_id,
        
        case 
            when layer_number is null then true
            else false
        end as missing_layer_number,
        
        case 
            when material_type is null then true
            else false
        end as missing_material_type,
        
        -- Data validation flags
        case 
            when uniformity_score < 0 or uniformity_score > 100 then true
            else false
        end as invalid_uniformity_score,
        
        case 
            when coverage_percentage < 0 or coverage_percentage > 100 then true
            else false
        end as invalid_coverage_percentage,
        
        case 
            when thickness_consistency < 0 or thickness_consistency > 100 then true
            else false
        end as invalid_thickness_consistency,
        
        case 
            when surface_roughness is not null and (surface_roughness < 0 or surface_roughness > 100) then true
            else false
        end as invalid_surface_roughness,
        
        case 
            when powder_density is not null and (powder_density < 0.1 or powder_density > 20) then true
            else false
        end as invalid_powder_density,
        
        case 
            when flowability is not null and (flowability < 0 or flowability > 100) then true
            else false
        end as invalid_flowability,
        
        case 
            when moisture_content is not null and (moisture_content < 0 or moisture_content > 100) then true
            else false
        end as invalid_moisture_content,
        
        -- Quality grade (numeric)
        case 
            when overall_quality_assessment = 'EXCELLENT' then 100
            when overall_quality_assessment = 'GOOD' then 80
            when overall_quality_assessment = 'ACCEPTABLE' then 60
            when overall_quality_assessment = 'POOR' then 40
            when overall_quality_assessment = 'UNACCEPTABLE' then 20
            else 0
        end as quality_grade_score,
        
        -- Overall quality score calculation
        (
            uniformity_score * 0.3 +
            coverage_percentage * 0.3 +
            thickness_consistency * 0.2 +
            case when surface_roughness is not null then 
                greatest(0, 100 - surface_roughness) * 0.1 else 0 end +
            case when density_variation is not null then 
                greatest(0, 100 - density_variation * 100) * 0.1 else 0 end
        ) as calculated_quality_score,
        
        -- Image quality score
        case 
            when brightness is not null and contrast is not null and sharpness is not null then
                (
                    (100 - abs(brightness - 150) / 1.5) * 0.2 +
                    least(100, contrast * 2) * 0.25 +
                    least(100, sharpness * 1.3) * 0.25 +
                    (100 - abs(red_channel - green_channel) / 2.55) * 0.2
                ) - least(30, coalesce(noise_level, 0) * 10)
            else null
        end as image_quality_score,
        
        -- Powder flowability score
        coalesce(flowability, 0) as powder_flowability_score,
        
        -- Moisture risk score
        case 
            when moisture_content is null then 0
            when moisture_content < 0.1 then 0
            when moisture_content < 0.5 then moisture_content * 20
            when moisture_content < 1.0 then 10 + (moisture_content - 0.5) * 40
            else 30 + least(70, (moisture_content - 1.0) * 70)
        end as moisture_risk_score,
        
        -- Environmental risk score
        (
            case when ambient_temperature < 15 or ambient_temperature > 30 then 
                least(30, abs(ambient_temperature - 22.5) * 3) else 0 end +
            case when relative_humidity < 30 or relative_humidity > 50 then 
                least(40, abs(relative_humidity - 40) * 2) else 0 end +
            case when vibration_level is not null and vibration_level > 0.1 then 
                least(30, vibration_level * 100) else 0 end
        ) as environmental_risk_score,
        
        -- File size in MB
        file_size / (1024.0 * 1024.0) as file_size_mb,
        
        -- Particle size distribution span
        case 
            when particle_size_d10 is not null and particle_size_d50 is not null and particle_size_d90 is not null then
                (particle_size_d90 - particle_size_d10) / particle_size_d50
            else null
        end as calculated_particle_span,
        
        -- Data completeness score
        (
            case when bed_id is not null then 1 else 0 end +
            case when process_id is not null then 1 else 0 end +
            case when layer_number is not null then 1 else 0 end +
            case when timestamp is not null then 1 else 0 end +
            case when material_type is not null then 1 else 0 end +
            case when uniformity_score is not null then 1 else 0 end +
            case when coverage_percentage is not null then 1 else 0 end +
            case when thickness_consistency is not null then 1 else 0 end +
            case when processing_status is not null then 1 else 0 end
        ) / 9.0 as data_completeness_score

    from source_data
),

final as (
    select
        *,
        -- Add row number for deduplication
        row_number() over (
            partition by bed_id 
            order by created_at desc
        ) as row_num
        
    from cleaned_data
)

select * from final
where row_num = 1  -- Keep only the latest record for each bed_id
