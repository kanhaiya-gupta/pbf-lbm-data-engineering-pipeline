-- Intermediate model for process parameters
-- This model aggregates and analyzes process parameters across different data sources

{{ config(
    materialized='table',
    tags=['intermediate', 'process_parameters']
) }}

with pbf_process_data as (
    select
        process_id,
        machine_id,
        build_id,
        timestamp,
        layer_number,
        temperature,
        pressure,
        laser_power,
        scan_speed,
        layer_height,
        hatch_spacing,
        exposure_time,
        atmosphere,
        powder_material,
        powder_batch_id,
        process_efficiency,
        estimated_energy_consumption_kwh,
        quality_grade,
        data_completeness_score
    from {{ ref('stg_pbf_process_data') }}
),

ispm_monitoring_data as (
    select
        process_id,
        sensor_type,
        measurement_value,
        unit,
        signal_quality_score,
        measurement_confidence,
        risk_score,
        timestamp as monitoring_timestamp
    from {{ ref('stg_ispm_monitoring_data') }}
),

-- Aggregate ISPM data by process and sensor type
ispm_aggregated as (
    select
        process_id,
        sensor_type,
        avg(measurement_value) as avg_measurement_value,
        min(measurement_value) as min_measurement_value,
        max(measurement_value) as max_measurement_value,
        stddev(measurement_value) as stddev_measurement_value,
        avg(signal_quality_score) as avg_signal_quality,
        avg(measurement_confidence) as avg_measurement_confidence,
        max(risk_score) as max_risk_score,
        count(*) as measurement_count,
        min(monitoring_timestamp) as first_measurement,
        max(monitoring_timestamp) as last_measurement
    from ispm_monitoring_data
    group by process_id, sensor_type
),

-- Pivot ISPM data to get sensor measurements as columns
ispm_pivoted as (
    select
        process_id,
        max(case when sensor_type = 'THERMAL' then avg_measurement_value end) as avg_thermal_measurement,
        max(case when sensor_type = 'THERMAL' then stddev_measurement_value end) as thermal_measurement_stddev,
        max(case when sensor_type = 'THERMAL' then avg_signal_quality end) as thermal_signal_quality,
        max(case when sensor_type = 'THERMAL' then max_risk_score end) as thermal_risk_score,
        
        max(case when sensor_type = 'OPTICAL' then avg_measurement_value end) as avg_optical_measurement,
        max(case when sensor_type = 'OPTICAL' then stddev_measurement_value end) as optical_measurement_stddev,
        max(case when sensor_type = 'OPTICAL' then avg_signal_quality end) as optical_signal_quality,
        max(case when sensor_type = 'OPTICAL' then max_risk_score end) as optical_risk_score,
        
        max(case when sensor_type = 'ACOUSTIC' then avg_measurement_value end) as avg_acoustic_measurement,
        max(case when sensor_type = 'ACOUSTIC' then stddev_measurement_value end) as acoustic_measurement_stddev,
        max(case when sensor_type = 'ACOUSTIC' then avg_signal_quality end) as acoustic_signal_quality,
        max(case when sensor_type = 'ACOUSTIC' then max_risk_score end) as acoustic_risk_score,
        
        max(case when sensor_type = 'VIBRATION' then avg_measurement_value end) as avg_vibration_measurement,
        max(case when sensor_type = 'VIBRATION' then stddev_measurement_value end) as vibration_measurement_stddev,
        max(case when sensor_type = 'VIBRATION' then avg_signal_quality end) as vibration_signal_quality,
        max(case when sensor_type = 'VIBRATION' then max_risk_score end) as vibration_risk_score,
        
        max(case when sensor_type = 'PRESSURE' then avg_measurement_value end) as avg_pressure_measurement,
        max(case when sensor_type = 'PRESSURE' then stddev_measurement_value end) as pressure_measurement_stddev,
        max(case when sensor_type = 'PRESSURE' then avg_signal_quality end) as pressure_signal_quality,
        max(case when sensor_type = 'PRESSURE' then max_risk_score end) as pressure_risk_score,
        
        max(case when sensor_type = 'MELT_POOL' then avg_measurement_value end) as avg_melt_pool_measurement,
        max(case when sensor_type = 'MELT_POOL' then stddev_measurement_value end) as melt_pool_measurement_stddev,
        max(case when sensor_type = 'MELT_POOL' then avg_signal_quality end) as melt_pool_signal_quality,
        max(case when sensor_type = 'MELT_POOL' then max_risk_score end) as melt_pool_risk_score,
        
        max(case when sensor_type = 'LAYER_HEIGHT' then avg_measurement_value end) as avg_layer_height_measurement,
        max(case when sensor_type = 'LAYER_HEIGHT' then stddev_measurement_value end) as layer_height_measurement_stddev,
        max(case when sensor_type = 'LAYER_HEIGHT' then avg_signal_quality end) as layer_height_signal_quality,
        max(case when sensor_type = 'LAYER_HEIGHT' then max_risk_score end) as layer_height_risk_score,
        
        -- Overall ISPM metrics
        avg(avg_signal_quality) as overall_signal_quality,
        avg(avg_measurement_confidence) as overall_measurement_confidence,
        max(max_risk_score) as overall_max_risk_score,
        sum(measurement_count) as total_measurements
        
    from ispm_aggregated
    group by process_id
),

-- Combine PBF process data with ISPM data
combined_data as (
    select
        p.process_id,
        p.machine_id,
        p.build_id,
        p.timestamp,
        p.layer_number,
        
        -- Core process parameters
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
        
        -- Calculated metrics
        p.process_efficiency,
        p.estimated_energy_consumption_kwh,
        p.quality_grade,
        p.data_completeness_score,
        
        -- ISPM sensor data
        i.avg_thermal_measurement,
        i.thermal_measurement_stddev,
        i.thermal_signal_quality,
        i.thermal_risk_score,
        
        i.avg_optical_measurement,
        i.optical_measurement_stddev,
        i.optical_signal_quality,
        i.optical_risk_score,
        
        i.avg_acoustic_measurement,
        i.acoustic_measurement_stddev,
        i.acoustic_signal_quality,
        i.acoustic_risk_score,
        
        i.avg_vibration_measurement,
        i.vibration_measurement_stddev,
        i.vibration_signal_quality,
        i.vibration_risk_score,
        
        i.avg_pressure_measurement,
        i.pressure_measurement_stddev,
        i.pressure_signal_quality,
        i.pressure_risk_score,
        
        i.avg_melt_pool_measurement,
        i.melt_pool_measurement_stddev,
        i.melt_pool_signal_quality,
        i.melt_pool_risk_score,
        
        i.avg_layer_height_measurement,
        i.layer_height_measurement_stddev,
        i.layer_height_signal_quality,
        i.layer_height_risk_score,
        
        -- Overall ISPM metrics
        i.overall_signal_quality,
        i.overall_measurement_confidence,
        i.overall_max_risk_score,
        i.total_measurements,
        
        -- Parameter stability analysis
        case 
            when i.thermal_measurement_stddev is not null then
                case 
                    when i.thermal_measurement_stddev < 5 then 'STABLE'
                    when i.thermal_measurement_stddev < 15 then 'MODERATE'
                    else 'UNSTABLE'
                end
            else 'UNKNOWN'
        end as thermal_stability,
        
        case 
            when i.pressure_measurement_stddev is not null then
                case 
                    when i.pressure_measurement_stddev < 0.1 then 'STABLE'
                    when i.pressure_measurement_stddev < 0.5 then 'MODERATE'
                    else 'UNSTABLE'
                end
            else 'UNKNOWN'
        end as pressure_stability,
        
        case 
            when i.laser_power is not null and i.avg_optical_measurement is not null then
                case 
                    when abs(i.laser_power - i.avg_optical_measurement) < 10 then 'ALIGNED'
                    when abs(i.laser_power - i.avg_optical_measurement) < 50 then 'MODERATE'
                    else 'MISALIGNED'
                end
            else 'UNKNOWN'
        end as laser_optical_alignment,
        
        -- Process parameter optimization score
        (
            case when p.process_efficiency > 0.8 then 25 else p.process_efficiency * 31.25 end +
            case when i.overall_signal_quality > 80 then 25 else i.overall_signal_quality * 0.3125 end +
            case when i.overall_max_risk_score < 20 then 25 else (100 - i.overall_max_risk_score) * 0.3125 end +
            case when p.data_completeness_score > 0.9 then 25 else p.data_completeness_score * 27.78 end
        ) as parameter_optimization_score
        
    from pbf_process_data p
    left join ispm_pivoted i on p.process_id = i.process_id
)

select * from combined_data
