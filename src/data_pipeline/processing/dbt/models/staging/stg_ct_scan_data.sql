-- Staging model for CT scan data
-- This model cleans and standardizes raw CT scan data

{{ config(
    materialized='view',
    tags=['staging', 'ct_scan']
) }}

with source_data as (
    select * from {{ source('raw_data', 'ct_scan_data') }}
),

cleaned_data as (
    select
        -- Primary identifiers
        scan_id,
        process_id,
        part_id,
        
        -- Timestamps
        created_at,
        updated_at,
        
        -- Scan information
        scan_type,
        processing_status,
        
        -- Scan parameters
        voltage,
        current,
        exposure_time,
        number_of_projections,
        detector_resolution,
        voxel_size,
        scan_duration,
        
        -- File metadata
        file_path,
        file_format,
        file_size,
        compression,
        checksum,
        
        -- Image dimensions
        image_width,
        image_height,
        image_depth,
        physical_width,
        physical_height,
        physical_depth,
        
        -- Quality metrics
        contrast_to_noise_ratio,
        signal_to_noise_ratio,
        spatial_resolution,
        uniformity,
        artifacts_detected,
        artifact_severity,
        
        -- Defect analysis
        total_defects,
        overall_quality_score,
        acceptance_status,
        
        -- Dimensional analysis
        dimensional_accuracy,
        
        -- Processing metadata
        processing_metadata,
        metadata,
        
        -- Data quality flags
        case 
            when scan_id is null then true
            else false
        end as missing_scan_id,
        
        case 
            when process_id is null then true
            else false
        end as missing_process_id,
        
        case 
            when scan_type is null then true
            else false
        end as missing_scan_type,
        
        case 
            when processing_status is null then true
            else false
        end as missing_processing_status,
        
        -- Data validation flags
        case 
            when voltage < 10 or voltage > 500 then true
            else false
        end as invalid_voltage,
        
        case 
            when current < 0.1 or current > 1000 then true
            else false
        end as invalid_current,
        
        case 
            when exposure_time < 0.001 or exposure_time > 60 then true
            else false
        end as invalid_exposure_time,
        
        case 
            when number_of_projections < 100 or number_of_projections > 10000 then true
            else false
        end as invalid_projections,
        
        case 
            when voxel_size < 0.1 or voxel_size > 1000 then true
            else false
        end as invalid_voxel_size,
        
        case 
            when file_size <= 0 then true
            else false
        end as invalid_file_size,
        
        case 
            when image_width <= 0 or image_height <= 0 or image_depth <= 0 then true
            else false
        end as invalid_image_dimensions,
        
        -- Quality assessment
        case 
            when overall_quality_score >= 90 and total_defects = 0 and 
                 (artifacts_detected = false or artifact_severity = 'NONE') then 'EXCELLENT'
            when overall_quality_score >= 80 and total_defects <= 5 and 
                 (artifacts_detected = false or artifact_severity in ('NONE', 'MINIMAL')) then 'GOOD'
            when overall_quality_score >= 70 and total_defects <= 10 and 
                 artifact_severity in ('NONE', 'MINIMAL', 'MODERATE') then 'ACCEPTABLE'
            else 'POOR'
        end as quality_grade,
        
        -- Scan quality score calculation
        (
            case when contrast_to_noise_ratio is not null then 
                least(100, (contrast_to_noise_ratio / 10.0) * 100) else 0 end +
            case when signal_to_noise_ratio is not null then 
                least(100, (signal_to_noise_ratio / 20.0) * 100) else 0 end +
            case when spatial_resolution is not null then 
                least(100, (spatial_resolution / 5.0) * 100) else 0 end +
            case when uniformity is not null then uniformity else 0 end
        ) / 4.0 as calculated_quality_score,
        
        -- Artifact penalty
        case 
            when artifacts_detected = true then
                case 
                    when artifact_severity = 'NONE' then 0
                    when artifact_severity = 'MINIMAL' then 5
                    when artifact_severity = 'MODERATE' then 15
                    when artifact_severity = 'SEVERE' then 30
                    else 0
                end
            else 0
        end as artifact_penalty,
        
        -- File size in MB
        file_size / (1024.0 * 1024.0) as file_size_mb,
        
        -- Voxel volume in mm³
        power(voxel_size / 1000.0, 3) as voxel_volume_mm3,
        
        -- Total scanned volume in mm³
        physical_width * physical_height * physical_depth as total_volume_mm3,
        
        -- Scan efficiency
        case 
            when scan_duration > 0 then
                least(1.0, (number_of_projections * exposure_time / 60.0) / scan_duration)
            else 0.0
        end as scan_efficiency,
        
        -- Data completeness score
        (
            case when scan_id is not null then 1 else 0 end +
            case when process_id is not null then 1 else 0 end +
            case when scan_type is not null then 1 else 0 end +
            case when processing_status is not null then 1 else 0 end +
            case when voltage is not null then 1 else 0 end +
            case when current is not null then 1 else 0 end +
            case when exposure_time is not null then 1 else 0 end +
            case when number_of_projections is not null then 1 else 0 end +
            case when voxel_size is not null then 1 else 0 end +
            case when file_size is not null then 1 else 0 end
        ) / 10.0 as data_completeness_score

    from source_data
),

final as (
    select
        *,
        -- Add row number for deduplication
        row_number() over (
            partition by scan_id 
            order by created_at desc
        ) as row_num
        
    from cleaned_data
)

select * from final
where row_num = 1  -- Keep only the latest record for each scan_id
