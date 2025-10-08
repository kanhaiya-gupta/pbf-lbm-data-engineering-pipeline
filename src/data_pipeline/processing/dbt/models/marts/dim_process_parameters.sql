-- Dimension table for process parameters
-- This model provides standardized process parameters for dimensional modeling

{{ config(
    materialized='table',
    tags=['marts', 'dimension', 'process_parameters']
) }}

with process_parameters as (
    select * from {{ ref('int_process_parameters') }}
),

-- Create dimension table with standardized parameter ranges and categories
dimension_data as (
    select
        process_id,
        machine_id,
        build_id,
        timestamp,
        layer_number,
        
        -- Core process parameters
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
        
        -- Parameter categories
        case 
            when temperature < 800 then 'LOW_TEMPERATURE'
            when temperature between 800 and 1200 then 'MEDIUM_TEMPERATURE'
            when temperature between 1200 and 1600 then 'HIGH_TEMPERATURE'
            when temperature > 1600 then 'VERY_HIGH_TEMPERATURE'
            else 'UNKNOWN_TEMPERATURE'
        end as temperature_category,
        
        case 
            when pressure < 0.1 then 'ULTRA_LOW_PRESSURE'
            when pressure between 0.1 and 1.0 then 'LOW_PRESSURE'
            when pressure between 1.0 and 10.0 then 'MEDIUM_PRESSURE'
            when pressure between 10.0 and 100.0 then 'HIGH_PRESSURE'
            when pressure > 100.0 then 'VERY_HIGH_PRESSURE'
            else 'UNKNOWN_PRESSURE'
        end as pressure_category,
        
        case 
            when laser_power < 50 then 'LOW_POWER'
            when laser_power between 50 and 150 then 'MEDIUM_POWER'
            when laser_power between 150 and 300 then 'HIGH_POWER'
            when laser_power between 300 and 500 then 'VERY_HIGH_POWER'
            when laser_power > 500 then 'ULTRA_HIGH_POWER'
            else 'UNKNOWN_POWER'
        end as laser_power_category,
        
        case 
            when scan_speed < 500 then 'LOW_SPEED'
            when scan_speed between 500 and 1500 then 'MEDIUM_SPEED'
            when scan_speed between 1500 and 3000 then 'HIGH_SPEED'
            when scan_speed between 3000 and 5000 then 'VERY_HIGH_SPEED'
            when scan_speed > 5000 then 'ULTRA_HIGH_SPEED'
            else 'UNKNOWN_SPEED'
        end as scan_speed_category,
        
        case 
            when layer_height < 0.02 then 'ULTRA_THIN_LAYER'
            when layer_height between 0.02 and 0.05 then 'THIN_LAYER'
            when layer_height between 0.05 and 0.1 then 'MEDIUM_LAYER'
            when layer_height between 0.1 and 0.2 then 'THICK_LAYER'
            when layer_height > 0.2 then 'VERY_THICK_LAYER'
            else 'UNKNOWN_LAYER_HEIGHT'
        end as layer_height_category,
        
        case 
            when hatch_spacing < 0.05 then 'FINE_HATCH'
            when hatch_spacing between 0.05 and 0.1 then 'MEDIUM_HATCH'
            when hatch_spacing between 0.1 and 0.2 then 'COARSE_HATCH'
            when hatch_spacing > 0.2 then 'VERY_COARSE_HATCH'
            else 'UNKNOWN_HATCH_SPACING'
        end as hatch_spacing_category,
        
        case 
            when exposure_time < 30 then 'SHORT_EXPOSURE'
            when exposure_time between 30 and 120 then 'MEDIUM_EXPOSURE'
            when exposure_time between 120 and 300 then 'LONG_EXPOSURE'
            when exposure_time > 300 then 'VERY_LONG_EXPOSURE'
            else 'UNKNOWN_EXPOSURE_TIME'
        end as exposure_time_category,
        
        -- Material categories
        case 
            when powder_material ilike '%ti6al4v%' or powder_material ilike '%titanium%' then 'TITANIUM_ALLOY'
            when powder_material ilike '%inconel%' or powder_material ilike '%nickel%' then 'NICKEL_ALLOY'
            when powder_material ilike '%stainless%' or powder_material ilike '%steel%' then 'STEEL_ALLOY'
            when powder_material ilike '%aluminum%' or powder_material ilike '%al%' then 'ALUMINUM_ALLOY'
            when powder_material ilike '%copper%' or powder_material ilike '%cu%' then 'COPPER_ALLOY'
            when powder_material ilike '%cobalt%' then 'COBALT_ALLOY'
            else 'OTHER_MATERIAL'
        end as material_category,
        
        -- Atmosphere categories
        case 
            when atmosphere = 'argon' then 'INERT_GAS'
            when atmosphere = 'nitrogen' then 'INERT_GAS'
            when atmosphere = 'helium' then 'INERT_GAS'
            when atmosphere = 'vacuum' then 'VACUUM'
            when atmosphere = 'air' then 'REACTIVE_ATMOSPHERE'
            else 'UNKNOWN_ATMOSPHERE'
        end as atmosphere_category,
        
        -- Parameter optimization levels
        case 
            when process_efficiency >= 0.9 then 'OPTIMIZED'
            when process_efficiency >= 0.8 then 'WELL_TUNED'
            when process_efficiency >= 0.7 then 'ADEQUATE'
            when process_efficiency >= 0.6 then 'NEEDS_OPTIMIZATION'
            else 'POORLY_OPTIMIZED'
        end as optimization_level,
        
        -- Parameter stability levels
        case 
            when thermal_stability = 'STABLE' and pressure_stability = 'STABLE' then 'HIGHLY_STABLE'
            when thermal_stability = 'STABLE' or pressure_stability = 'STABLE' then 'MODERATELY_STABLE'
            when thermal_stability = 'UNSTABLE' or pressure_stability = 'UNSTABLE' then 'UNSTABLE'
            else 'UNKNOWN_STABILITY'
        end as parameter_stability_level,
        
        -- Laser alignment status
        case 
            when laser_optical_alignment = 'ALIGNED' then 'PROPERLY_ALIGNED'
            when laser_optical_alignment = 'MODERATE' then 'PARTIALLY_ALIGNED'
            when laser_optical_alignment = 'MISALIGNED' then 'MISALIGNED'
            else 'UNKNOWN_ALIGNMENT'
        end as laser_alignment_status,
        
        -- Process complexity level
        case 
            when layer_height < 0.05 and hatch_spacing < 0.1 and scan_speed > 2000 then 'HIGH_COMPLEXITY'
            when layer_height between 0.05 and 0.1 and hatch_spacing between 0.1 and 0.2 and scan_speed between 1000 and 2000 then 'MEDIUM_COMPLEXITY'
            when layer_height > 0.1 and hatch_spacing > 0.2 and scan_speed < 1000 then 'LOW_COMPLEXITY'
            else 'VARIABLE_COMPLEXITY'
        end as process_complexity_level,
        
        -- Energy intensity level
        case 
            when laser_power > 300 and exposure_time > 120 then 'HIGH_ENERGY'
            when laser_power between 150 and 300 and exposure_time between 60 and 120 then 'MEDIUM_ENERGY'
            when laser_power < 150 and exposure_time < 60 then 'LOW_ENERGY'
            else 'VARIABLE_ENERGY'
        end as energy_intensity_level,
        
        -- Process speed category
        case 
            when scan_speed > 3000 and layer_height > 0.1 then 'FAST_BUILD'
            when scan_speed between 1000 and 3000 and layer_height between 0.05 and 0.1 then 'MEDIUM_BUILD'
            when scan_speed < 1000 and layer_height < 0.05 then 'SLOW_BUILD'
            else 'VARIABLE_SPEED'
        end as build_speed_category,
        
        -- Quality potential level
        case 
            when temperature between 1000 and 1400 and pressure between 0.5 and 5.0 and 
                 laser_power between 150 and 300 and scan_speed between 1000 and 2000 then 'HIGH_QUALITY_POTENTIAL'
            when temperature between 800 and 1600 and pressure between 0.1 and 10.0 and 
                 laser_power between 100 and 400 and scan_speed between 500 and 3000 then 'MEDIUM_QUALITY_POTENTIAL'
            else 'VARIABLE_QUALITY_POTENTIAL'
        end as quality_potential_level,
        
        -- Process efficiency metrics
        process_efficiency,
        estimated_energy_consumption_kwh,
        parameter_optimization_score,
        quality_grade,
        data_completeness_score,
        
        -- ISPM sensor data
        avg_thermal_measurement,
        thermal_measurement_stddev,
        thermal_signal_quality,
        thermal_risk_score,
        avg_optical_measurement,
        optical_measurement_stddev,
        optical_signal_quality,
        optical_risk_score,
        avg_acoustic_measurement,
        acoustic_measurement_stddev,
        acoustic_signal_quality,
        acoustic_risk_score,
        avg_vibration_measurement,
        vibration_measurement_stddev,
        vibration_signal_quality,
        vibration_risk_score,
        avg_pressure_measurement,
        pressure_measurement_stddev,
        pressure_signal_quality,
        pressure_risk_score,
        avg_melt_pool_measurement,
        melt_pool_measurement_stddev,
        melt_pool_signal_quality,
        melt_pool_risk_score,
        avg_layer_height_measurement,
        layer_height_measurement_stddev,
        layer_height_signal_quality,
        layer_height_risk_score,
        overall_signal_quality,
        overall_measurement_confidence,
        overall_max_risk_score,
        total_measurements,
        thermal_stability,
        pressure_stability,
        laser_optical_alignment
        
    from process_parameters
)

select
    *,
    
    -- Parameter combination score
    case 
        when temperature_category = 'MEDIUM_TEMPERATURE' and pressure_category = 'LOW_PRESSURE' and 
             laser_power_category = 'MEDIUM_POWER' and scan_speed_category = 'MEDIUM_SPEED' then 'OPTIMAL_COMBINATION'
        when temperature_category in ('MEDIUM_TEMPERATURE', 'HIGH_TEMPERATURE') and 
             pressure_category in ('LOW_PRESSURE', 'MEDIUM_PRESSURE') and 
             laser_power_category in ('MEDIUM_POWER', 'HIGH_POWER') then 'GOOD_COMBINATION'
        else 'STANDARD_COMBINATION'
    end as parameter_combination_quality,
    
    -- Process repeatability score
    case 
        when parameter_stability_level = 'HIGHLY_STABLE' and laser_alignment_status = 'PROPERLY_ALIGNED' then 'HIGH_REPEATABILITY'
        when parameter_stability_level = 'MODERATELY_STABLE' and laser_alignment_status in ('PROPERLY_ALIGNED', 'PARTIALLY_ALIGNED') then 'MEDIUM_REPEATABILITY'
        else 'LOW_REPEATABILITY'
    end as process_repeatability_score,
    
    -- Parameter validation status
    case 
        when temperature is not null and pressure is not null and laser_power is not null and 
             scan_speed is not null and layer_height is not null then 'COMPLETE_PARAMETERS'
        when temperature is not null and pressure is not null and laser_power is not null then 'CORE_PARAMETERS'
        else 'INCOMPLETE_PARAMETERS'
    end as parameter_completeness_status,
    
    -- Process maturity level
    case 
        when optimization_level = 'OPTIMIZED' and parameter_stability_level = 'HIGHLY_STABLE' and 
             process_repeatability_score = 'HIGH_REPEATABILITY' then 'MATURE_PROCESS'
        when optimization_level in ('OPTIMIZED', 'WELL_TUNED') and parameter_stability_level in ('HIGHLY_STABLE', 'MODERATELY_STABLE') then 'DEVELOPING_PROCESS'
        else 'EMERGING_PROCESS'
    end as process_maturity_level

from dimension_data
