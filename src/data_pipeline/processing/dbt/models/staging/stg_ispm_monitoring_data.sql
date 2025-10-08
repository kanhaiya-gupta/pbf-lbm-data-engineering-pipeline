-- Staging model for ISPM monitoring data
-- This model cleans and standardizes raw ISPM monitoring data

{{ config(
    materialized='view',
    tags=['staging', 'ispm_monitoring']
) }}

with source_data as (
    select * from {{ source('raw_data', 'ispm_monitoring_data') }}
),

cleaned_data as (
    select
        -- Primary identifiers
        monitoring_id,
        process_id,
        sensor_id,
        
        -- Timestamps
        timestamp,
        created_at,
        updated_at,
        
        -- Sensor information
        sensor_type,
        sensor_location_x,
        sensor_location_y,
        sensor_location_z,
        
        -- Measurement data
        measurement_value,
        unit,
        measurement_range_min,
        measurement_range_max,
        measurement_accuracy,
        sampling_rate,
        
        -- Signal quality
        signal_quality,
        noise_level,
        
        -- Calibration information
        calibration_status,
        last_calibration_date,
        
        -- Environmental conditions
        ambient_temperature,
        relative_humidity,
        vibration_level,
        
        -- Anomaly detection
        anomaly_detected,
        anomaly_type,
        anomaly_severity,
        
        -- Additional data
        raw_data,
        processed_data,
        metadata,
        
        -- Data quality flags
        case 
            when measurement_value is null then true
            else false
        end as missing_measurement_value,
        
        case 
            when unit is null or unit = '' then true
            else false
        end as missing_unit,
        
        case 
            when sensor_type is null then true
            else false
        end as missing_sensor_type,
        
        -- Data validation flags
        case 
            when measurement_range_min is not null and measurement_range_max is not null then
                case 
                    when measurement_value < measurement_range_min or 
                         measurement_value > measurement_range_max then true
                    else false
                end
            else false
        end as measurement_out_of_range,
        
        case 
            when noise_level is not null and noise_level < 0 then true
            else false
        end as invalid_noise_level,
        
        case 
            when sampling_rate is not null and (sampling_rate < 0 or sampling_rate > 1000000) then true
            else false
        end as invalid_sampling_rate,
        
        -- Signal quality score (numeric)
        case 
            when signal_quality = 'EXCELLENT' then 100
            when signal_quality = 'GOOD' then 80
            when signal_quality = 'FAIR' then 60
            when signal_quality = 'POOR' then 40
            when signal_quality = 'UNKNOWN' then 0
            else 0
        end as signal_quality_score,
        
        -- Anomaly severity score (numeric)
        case 
            when anomaly_severity = 'CRITICAL' then 100
            when anomaly_severity = 'HIGH' then 70
            when anomaly_severity = 'MEDIUM' then 40
            when anomaly_severity = 'LOW' then 20
            else 0
        end as anomaly_severity_score,
        
        -- Measurement deviation from range center
        case 
            when measurement_range_min is not null and measurement_range_max is not null then
                abs(measurement_value - (measurement_range_min + measurement_range_max) / 2.0) / 
                (measurement_range_max - measurement_range_min) * 100
            else null
        end as measurement_deviation_percent,
        
        -- Risk score calculation
        (
            case when anomaly_detected then anomaly_severity_score else 0 end +
            case when signal_quality_score < 60 then (100 - signal_quality_score) * 0.3 else 0 end +
            case when measurement_out_of_range then 30 else 0 end +
            case when noise_level > 0.1 then least(20, noise_level * 100) else 0 end +
            case when calibration_status = false then 15 else 0 end
        ) as risk_score,
        
        -- Measurement confidence score
        (
            signal_quality_score / 100.0 *
            case when noise_level is not null then greatest(0, 1 - noise_level * 5) else 1 end *
            case when calibration_status = true then 1 else 0.8 end *
            case when measurement_out_of_range then 0.7 else 1 end
        ) as measurement_confidence,
        
        -- Data completeness score
        (
            case when monitoring_id is not null then 1 else 0 end +
            case when process_id is not null then 1 else 0 end +
            case when sensor_id is not null then 1 else 0 end +
            case when timestamp is not null then 1 else 0 end +
            case when sensor_type is not null then 1 else 0 end +
            case when measurement_value is not null then 1 else 0 end +
            case when unit is not null then 1 else 0 end
        ) / 7.0 as data_completeness_score

    from source_data
),

final as (
    select
        *,
        -- Add row number for deduplication
        row_number() over (
            partition by monitoring_id 
            order by created_at desc
        ) as row_num
        
    from cleaned_data
)

select * from final
where row_num = 1  -- Keep only the latest record for each monitoring_id
