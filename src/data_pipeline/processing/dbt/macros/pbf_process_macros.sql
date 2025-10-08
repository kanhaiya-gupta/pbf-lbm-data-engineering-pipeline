-- PBF Process Macros
-- Reusable SQL logic for PBF process data transformations

{% macro calculate_quality_score(density, surface_roughness, dimensional_accuracy, defect_count) %}
    -- Calculate overall quality score based on individual metrics
    case 
        when {{ density }} is not null and {{ surface_roughness }} is not null and 
             {{ dimensional_accuracy }} is not null and {{ defect_count }} is not null then
            (
                coalesce({{ density }}, 0) * 0.4 +
                coalesce(100 - {{ surface_roughness }}, 0) * 0.3 +
                coalesce(100 - ({{ dimensional_accuracy }} * 100), 0) * 0.2 +
                coalesce(greatest(0, 100 - ({{ defect_count }} * 10)), 0) * 0.1
            )
        else null
    end
{% endmacro %}

{% macro get_quality_grade(quality_score) %}
    -- Get quality grade based on quality score
    case 
        when {{ quality_score }} >= 90 then 'EXCELLENT'
        when {{ quality_score }} >= 80 then 'GOOD'
        when {{ quality_score }} >= 70 then 'ACCEPTABLE'
        when {{ quality_score }} >= 60 then 'POOR'
        else 'UNACCEPTABLE'
    end
{% endmacro %}

{% macro calculate_process_efficiency(scan_speed, layer_height) %}
    -- Calculate process efficiency based on scan speed and layer height
    case 
        when {{ scan_speed }} > 0 and {{ layer_height }} > 0 then
            least(1.0, ({{ scan_speed }} / 1000.0) * ({{ layer_height }} / 0.1))
        else 0.0
    end
{% endmacro %}

{% macro calculate_energy_consumption(laser_power, exposure_time) %}
    -- Calculate energy consumption in kWh
    case 
        when {{ laser_power }} is not null and {{ exposure_time }} is not null then
            ({{ laser_power }} * {{ exposure_time }}) / 3600000.0  -- Convert to kWh
        else null
    end
{% endmacro %}

{% macro validate_process_parameters(temperature, pressure, laser_power, scan_speed, layer_height) %}
    -- Validate process parameters against recommended ranges
    case 
        when {{ temperature }} < 800 or {{ temperature }} > 1600 then 'INVALID_TEMPERATURE'
        when {{ pressure }} < 0.1 or {{ pressure }} > 100 then 'INVALID_PRESSURE'
        when {{ laser_power }} < 50 or {{ laser_power }} > 500 then 'INVALID_LASER_POWER'
        when {{ scan_speed }} < 100 or {{ scan_speed }} > 5000 then 'INVALID_SCAN_SPEED'
        when {{ layer_height }} < 0.02 or {{ layer_height }} > 0.2 then 'INVALID_LAYER_HEIGHT'
        else 'VALID_PARAMETERS'
    end
{% endmacro %}

{% macro get_material_recommendations(material_type) %}
    -- Get recommended parameter ranges for material type
    case 
        when lower({{ material_type }}) like '%ti6al4v%' or lower({{ material_type }}) like '%titanium%' then
            '{"temperature": {"min": 1000, "max": 1400, "optimal": 1200}, "laser_power": {"min": 150, "max": 300, "optimal": 200}}'
        when lower({{ material_type }}) like '%inconel%' or lower({{ material_type }}) like '%nickel%' then
            '{"temperature": {"min": 1200, "max": 1600, "optimal": 1400}, "laser_power": {"min": 200, "max": 400, "optimal": 250}}'
        when lower({{ material_type }}) like '%stainless%' or lower({{ material_type }}) like '%steel%' then
            '{"temperature": {"min": 800, "max": 1200, "optimal": 1000}, "laser_power": {"min": 100, "max": 250, "optimal": 180}}'
        else
            '{"temperature": {"min": 800, "max": 1600, "optimal": 1200}, "laser_power": {"min": 50, "max": 500, "optimal": 200}}'
    end
{% endmacro %}

{% macro calculate_parameter_optimization_score(process_efficiency, signal_quality, risk_score, data_completeness) %}
    -- Calculate parameter optimization score
    (
        coalesce({{ process_efficiency }}, 0) * 0.3 +
        coalesce({{ signal_quality }}, 0) * 0.3 +
        coalesce(100 - {{ risk_score }}, 0) * 0.2 +
        coalesce({{ data_completeness }}, 0) * 100 * 0.2
    )
{% endmacro %}

{% macro get_process_status(quality_score, defect_rate, efficiency_score) %}
    -- Get overall process status
    case 
        when {{ quality_score }} >= 90 and {{ defect_rate }} <= 2 and {{ efficiency_score }} >= 0.8 then 'OPTIMAL'
        when {{ quality_score }} >= 80 and {{ defect_rate }} <= 5 and {{ efficiency_score }} >= 0.7 then 'SATISFACTORY'
        when {{ quality_score }} >= 70 and {{ defect_rate }} <= 10 and {{ efficiency_score }} >= 0.6 then 'NEEDS_IMPROVEMENT'
        when {{ quality_score }} >= 60 and {{ defect_rate }} <= 20 and {{ efficiency_score }} >= 0.5 then 'REQUIRES_ATTENTION'
        else 'CRITICAL_ISSUES'
    end
{% endmacro %}

{% macro calculate_data_completeness_score(required_fields) %}
    -- Calculate data completeness score for required fields
    (
        {% for field in required_fields %}
        case when {{ field }} is not null then 1 else 0 end +
        {% endfor %}
        0
    ) / {{ required_fields | length }}.0
{% endmacro %}

{% macro get_optimization_recommendation(process_efficiency, quality_score, defect_rate, consistency_score, risk_score) %}
    -- Get process optimization recommendations
    case 
        when {{ process_efficiency }} < 0.7 then 'OPTIMIZE_PROCESS_PARAMETERS'
        when {{ quality_score }} < 70 then 'IMPROVE_QUALITY_CONTROL'
        when {{ defect_rate }} > 10 then 'ADDRESS_DEFECT_ISSUES'
        when {{ consistency_score }} < 70 then 'IMPROVE_PROCESS_STABILITY'
        when {{ risk_score }} > 50 then 'REDUCE_RISK_FACTORS'
        else 'MAINTAIN_CURRENT_SETTINGS'
    end
{% endmacro %}

{% macro calculate_layer_processing_rate(total_layers, duration_hours) %}
    -- Calculate layer processing rate
    case 
        when {{ total_layers }} is not null and {{ duration_hours }} is not null and {{ duration_hours }} > 0 then
            {{ total_layers }} / {{ duration_hours }}
        else null
    end
{% endmacro %}

{% macro get_parameter_stability_level(thermal_stability, pressure_stability) %}
    -- Get parameter stability level
    case 
        when {{ thermal_stability }} = 'STABLE' and {{ pressure_stability }} = 'STABLE' then 'HIGHLY_STABLE'
        when {{ thermal_stability }} = 'STABLE' or {{ pressure_stability }} = 'STABLE' then 'MODERATELY_STABLE'
        when {{ thermal_stability }} = 'UNSTABLE' or {{ pressure_stability }} = 'UNSTABLE' then 'UNSTABLE'
        else 'UNKNOWN_STABILITY'
    end
{% endmacro %}

{% macro calculate_quality_yield_percentage(excellent_layers, good_layers, acceptable_layers, total_layers) %}
    -- Calculate quality yield percentage
    case 
        when {{ total_layers }} is not null and {{ total_layers }} > 0 then
            (({{ excellent_layers }} + {{ good_layers }} + {{ acceptable_layers }})::float / {{ total_layers }}) * 100
        else null
    end
{% endmacro %}

{% macro get_defect_severity_classification(defect_count) %}
    -- Classify defect severity based on count
    case 
        when {{ defect_count }} = 0 then 'NO_DEFECTS'
        when {{ defect_count }} = 1 then 'SINGLE_DEFECT'
        when {{ defect_count }} between 2 and 5 then 'MODERATE_DEFECTS'
        when {{ defect_count }} between 6 and 10 then 'HIGH_DEFECTS'
        when {{ defect_count }} > 10 then 'CRITICAL_DEFECTS'
        else 'UNKNOWN_DEFECTS'
    end
{% endmacro %}

{% macro calculate_process_duration_hours(start_timestamp, end_timestamp) %}
    -- Calculate process duration in hours
    case 
        when {{ start_timestamp }} is not null and {{ end_timestamp }} is not null then
            extract(epoch from ({{ end_timestamp }} - {{ start_timestamp }})) / 3600.0
        else null
    end
{% endmacro %}

{% macro get_material_category(material_type) %}
    -- Categorize material type
    case 
        when lower({{ material_type }}) like '%ti6al4v%' or lower({{ material_type }}) like '%titanium%' then 'TITANIUM_ALLOY'
        when lower({{ material_type }}) like '%inconel%' or lower({{ material_type }}) like '%nickel%' then 'NICKEL_ALLOY'
        when lower({{ material_type }}) like '%stainless%' or lower({{ material_type }}) like '%steel%' then 'STEEL_ALLOY'
        when lower({{ material_type }}) like '%aluminum%' or lower({{ material_type }}) like '%al%' then 'ALUMINUM_ALLOY'
        when lower({{ material_type }}) like '%copper%' or lower({{ material_type }}) like '%cu%' then 'COPPER_ALLOY'
        when lower({{ material_type }}) like '%cobalt%' then 'COBALT_ALLOY'
        else 'OTHER_MATERIAL'
    end
{% endmacro %}

{% macro get_atmosphere_category(atmosphere) %}
    -- Categorize atmosphere type
    case 
        when {{ atmosphere }} = 'argon' then 'INERT_GAS'
        when {{ atmosphere }} = 'nitrogen' then 'INERT_GAS'
        when {{ atmosphere }} = 'helium' then 'INERT_GAS'
        when {{ atmosphere }} = 'vacuum' then 'VACUUM'
        when {{ atmosphere }} = 'air' then 'REACTIVE_ATMOSPHERE'
        else 'UNKNOWN_ATMOSPHERE'
    end
{% endmacro %}

{% macro calculate_cost_per_layer(energy_per_layer_kwh, cost_per_kwh) %}
    -- Calculate cost per layer
    case 
        when {{ energy_per_layer_kwh }} is not null then
            {{ energy_per_layer_kwh }} * coalesce({{ cost_per_kwh }}, 0.12)  -- Default $0.12 per kWh
        else null
    end
{% endmacro %}
