-- Staging model for PBF process data
-- This model cleans and standardizes raw PBF process data

{{ config(
    materialized='view',
    tags=['staging', 'pbf_process']
) }}

with source_data as (
    select * from {{ source('raw_data', 'pbf_process_data') }}
),

cleaned_data as (
    select
        -- Primary identifiers
        process_id,
        machine_id,
        build_id,
        
        -- Timestamps
        timestamp,
        created_at,
        updated_at,
        
        -- Process parameters
        layer_number,
        temperature,
        pressure,
        laser_power,
        scan_speed,
        layer_height,
        hatch_spacing,
        exposure_time,
        
        -- Material and environment
        atmosphere,
        powder_material,
        powder_batch_id,
        
        -- Quality metrics
        density,
        surface_roughness,
        dimensional_accuracy,
        defect_count,
        
        -- Additional data
        process_parameters,
        metadata,
        
        -- Data quality flags
        case 
            when temperature is null then true
            else false
        end as missing_temperature,
        
        case 
            when pressure is null then true
            else false
        end as missing_pressure,
        
        case 
            when laser_power is null then true
            else false
        end as missing_laser_power,
        
        case 
            when scan_speed is null then true
            else false
        end as missing_scan_speed,
        
        case 
            when layer_height is null then true
            else false
        end as missing_layer_height,
        
        -- Data validation flags
        case 
            when temperature < 0 or temperature > 2000 then true
            else false
        end as invalid_temperature,
        
        case 
            when pressure < 0 or pressure > 1000 then true
            else false
        end as invalid_pressure,
        
        case 
            when laser_power < 0 or laser_power > 1000 then true
            else false
        end as invalid_laser_power,
        
        case 
            when scan_speed < 0 or scan_speed > 10000 then true
            else false
        end as invalid_scan_speed,
        
        case 
            when layer_height < 0.01 or layer_height > 1.0 then true
            else false
        end as invalid_layer_height,
        
        -- Quality assessment
        case 
            when density >= 95 and surface_roughness <= 10 and 
                 dimensional_accuracy <= 0.1 and defect_count = 0 then 'EXCELLENT'
            when density >= 90 and surface_roughness <= 20 and 
                 dimensional_accuracy <= 0.2 and defect_count <= 5 then 'GOOD'
            when density >= 85 and surface_roughness <= 30 and 
                 dimensional_accuracy <= 0.5 and defect_count <= 10 then 'ACCEPTABLE'
            else 'POOR'
        end as quality_grade,
        
        -- Process efficiency score
        case 
            when scan_speed > 0 and layer_height > 0 then
                least(1.0, (scan_speed / 1000.0) * (layer_height / 0.1))
            else 0.0
        end as process_efficiency,
        
        -- Energy consumption estimate
        case 
            when laser_power is not null and exposure_time is not null then
                (laser_power * exposure_time) / 3600000.0  -- Convert to kWh
            else null
        end as estimated_energy_consumption_kwh,
        
        -- Data completeness score
        (
            case when process_id is not null then 1 else 0 end +
            case when machine_id is not null then 1 else 0 end +
            case when timestamp is not null then 1 else 0 end +
            case when temperature is not null then 1 else 0 end +
            case when pressure is not null then 1 else 0 end +
            case when laser_power is not null then 1 else 0 end +
            case when scan_speed is not null then 1 else 0 end +
            case when layer_height is not null then 1 else 0 end
        ) / 8.0 as data_completeness_score

    from source_data
),

final as (
    select
        *,
        -- Add row number for deduplication
        row_number() over (
            partition by process_id, timestamp 
            order by created_at desc
        ) as row_num
        
    from cleaned_data
)

select * from final
where row_num = 1  -- Keep only the latest record for each process_id, timestamp combination
