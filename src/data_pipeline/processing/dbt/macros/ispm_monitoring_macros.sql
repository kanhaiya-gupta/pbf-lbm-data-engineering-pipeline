-- ISPM Monitoring Macros
-- Reusable SQL logic for ISPM monitoring data transformations

{% macro calculate_signal_quality_score(signal_quality) %}
    -- Convert signal quality enum to numeric score
    case 
        when {{ signal_quality }} = 'EXCELLENT' then 100
        when {{ signal_quality }} = 'GOOD' then 80
        when {{ signal_quality }} = 'FAIR' then 60
        when {{ signal_quality }} = 'POOR' then 40
        when {{ signal_quality }} = 'UNKNOWN' then 0
        else 0
    end
{% endmacro %}

{% macro calculate_anomaly_severity_score(anomaly_severity) %}
    -- Convert anomaly severity enum to numeric score
    case 
        when {{ anomaly_severity }} = 'CRITICAL' then 100
        when {{ anomaly_severity }} = 'HIGH' then 70
        when {{ anomaly_severity }} = 'MEDIUM' then 40
        when {{ anomaly_severity }} = 'LOW' then 20
        else 0
    end
{% endmacro %}

{% macro calculate_measurement_deviation(measurement_value, min_value, max_value) %}
    -- Calculate deviation from expected range center
    case 
        when {{ min_value }} is not null and {{ max_value }} is not null then
            abs({{ measurement_value }} - ({{ min_value }} + {{ max_value }}) / 2.0) / 
            ({{ max_value }} - {{ min_value }}) * 100
        else null
    end
{% endmacro %}

{% macro is_measurement_in_range(measurement_value, min_value, max_value) %}
    -- Check if measurement is within expected range
    case 
        when {{ min_value }} is not null and {{ max_value }} is not null then
            case 
                when {{ measurement_value }} >= {{ min_value }} and {{ measurement_value }} <= {{ max_value }} then true
                else false
            end
        else true  -- No range specified, assume valid
    end
{% endmacro %}

{% macro calculate_risk_score(anomaly_detected, anomaly_severity, signal_quality, measurement_out_of_range, noise_level, calibration_status) %}
    -- Calculate overall risk score
    (
        case when {{ anomaly_detected }} then 
            case 
                when {{ anomaly_severity }} = 'CRITICAL' then 100
                when {{ anomaly_severity }} = 'HIGH' then 70
                when {{ anomaly_severity }} = 'MEDIUM' then 40
                when {{ anomaly_severity }} = 'LOW' then 20
                else 0
            end
        else 0 end +
        case when {{ signal_quality }} < 60 then (100 - {{ signal_quality }}) * 0.3 else 0 end +
        case when {{ measurement_out_of_range }} then 30 else 0 end +
        case when {{ noise_level }} > 0.1 then least(20, {{ noise_level }} * 100) else 0 end +
        case when {{ calibration_status }} = false then 15 else 0 end
    )
{% endmacro %}

{% macro calculate_measurement_confidence(signal_quality, noise_level, calibration_status, measurement_out_of_range) %}
    -- Calculate measurement confidence score
    (
        {{ signal_quality }} / 100.0 *
        case when {{ noise_level }} is not null then greatest(0, 1 - {{ noise_level }} * 5) else 1 end *
        case when {{ calibration_status }} = true then 1 else 0.8 end *
        case when {{ measurement_out_of_range }} then 0.7 else 1 end
    )
{% endmacro %}

{% macro get_sensor_type_category(sensor_type) %}
    -- Categorize sensor types
    case 
        when {{ sensor_type }} in ('THERMAL', 'OPTICAL', 'MELT_POOL') then 'PROCESS_MONITORING'
        when {{ sensor_type }} in ('ACOUSTIC', 'VIBRATION') then 'MECHANICAL_MONITORING'
        when {{ sensor_type }} in ('PRESSURE', 'GAS_ANALYSIS') then 'ENVIRONMENTAL_MONITORING'
        when {{ sensor_type }} = 'LAYER_HEIGHT' then 'GEOMETRIC_MONITORING'
        else 'OTHER_MONITORING'
    end
{% endmacro %}

{% macro get_anomaly_risk_level(risk_score) %}
    -- Get anomaly risk level based on risk score
    case 
        when {{ risk_score }} >= 80 then 'CRITICAL_RISK'
        when {{ risk_score }} >= 60 then 'HIGH_RISK'
        when {{ risk_score }} >= 40 then 'MODERATE_RISK'
        when {{ risk_score }} >= 20 then 'LOW_RISK'
        else 'NO_RISK'
    end
{% endmacro %}

{% macro get_recommended_action(risk_score, anomaly_detected, anomaly_severity) %}
    -- Get recommended action based on monitoring data
    case 
        when {{ anomaly_detected }} = true and {{ anomaly_severity }} = 'CRITICAL' then 'IMMEDIATE_ATTENTION_REQUIRED'
        when {{ risk_score }} > 70 then 'INVESTIGATE_ANOMALY'
        when {{ risk_score }} > 40 then 'MONITOR_CLOSELY'
        when {{ risk_score }} > 20 then 'CONTINUE_MONITORING'
        else 'NORMAL_OPERATION'
    end
{% endmacro %}

{% macro calculate_sensor_performance_score(signal_quality, measurement_confidence, calibration_status, noise_level) %}
    -- Calculate overall sensor performance score
    (
        {{ signal_quality }} * 0.4 +
        {{ measurement_confidence }} * 100 * 0.3 +
        case when {{ calibration_status }} = true then 100 else 0 end * 0.2 +
        case when {{ noise_level }} is not null then greatest(0, 100 - {{ noise_level }} * 100) else 100 end * 0.1
    )
{% endmacro %}

{% macro get_sensor_health_status(performance_score, calibration_status, last_calibration_date) %}
    -- Get sensor health status
    case 
        when {{ performance_score }} >= 90 and {{ calibration_status }} = true then 'EXCELLENT'
        when {{ performance_score }} >= 80 and {{ calibration_status }} = true then 'GOOD'
        when {{ performance_score }} >= 70 then 'FAIR'
        when {{ performance_score }} >= 60 then 'POOR'
        else 'CRITICAL'
    end
{% endmacro %}

{% macro calculate_sensor_uptime_score(measurement_count, expected_count) %}
    -- Calculate sensor uptime score
    case 
        when {{ expected_count }} > 0 then
            least(100, ({{ measurement_count }}::float / {{ expected_count }}) * 100)
        else 0
    end
{% endmacro %}

{% macro get_measurement_trend(measurement_values, window_size) %}
    -- Calculate measurement trend over a window
    case 
        when count(*) >= {{ window_size }} then
            case 
                when avg({{ measurement_values }}) > lag(avg({{ measurement_values }}), {{ window_size }}) over (order by timestamp) then 'INCREASING'
                when avg({{ measurement_values }}) < lag(avg({{ measurement_values }}), {{ window_size }}) over (order by timestamp) then 'DECREASING'
                else 'STABLE'
            end
        else 'INSUFFICIENT_DATA'
    end
{% endmacro %}

{% macro detect_measurement_anomalies(measurement_value, historical_mean, historical_stddev, threshold_multiplier) %}
    -- Detect measurement anomalies using statistical methods
    case 
        when {{ historical_stddev }} > 0 then
            case 
                when abs({{ measurement_value }} - {{ historical_mean }}) > ({{ historical_stddev }} * {{ threshold_multiplier }}) then true
                else false
            end
        else false
    end
{% endmacro %}

{% macro calculate_sensor_reliability_score(uptime_score, performance_score, calibration_status) %}
    -- Calculate sensor reliability score
    (
        {{ uptime_score }} * 0.4 +
        {{ performance_score }} * 0.4 +
        case when {{ calibration_status }} = true then 100 else 0 end * 0.2
    )
{% endmacro %}

{% macro get_sensor_maintenance_recommendation(reliability_score, last_calibration_date, performance_score) %}
    -- Get sensor maintenance recommendation
    case 
        when {{ reliability_score }} < 60 then 'IMMEDIATE_MAINTENANCE_REQUIRED'
        when {{ reliability_score }} < 80 then 'SCHEDULE_MAINTENANCE'
        when {{ last_calibration_date }} < current_date - interval '6 months' then 'RECALIBRATION_DUE'
        when {{ performance_score }} < 80 then 'PERFORMANCE_CHECK_REQUIRED'
        else 'MAINTENANCE_UP_TO_DATE'
    end
{% endmacro %}

{% macro calculate_environmental_impact_score(temperature, humidity, vibration_level) %}
    -- Calculate environmental impact score on sensor performance
    (
        case when {{ temperature }} < 15 or {{ temperature }} > 30 then 20 else 0 end +
        case when {{ humidity }} < 30 or {{ humidity }} > 60 then 15 else 0 end +
        case when {{ vibration_level }} > 0.1 then least(25, {{ vibration_level }} * 100) else 0 end
    )
{% endmacro %}

{% macro get_sensor_placement_quality_score(signal_quality, noise_level, measurement_confidence) %}
    -- Assess sensor placement quality
    case 
        when {{ signal_quality }} >= 80 and {{ noise_level }} < 0.05 and {{ measurement_confidence }} > 0.9 then 'OPTIMAL_PLACEMENT'
        when {{ signal_quality }} >= 60 and {{ noise_level }} < 0.1 and {{ measurement_confidence }} > 0.8 then 'GOOD_PLACEMENT'
        when {{ signal_quality }} >= 40 and {{ noise_level }} < 0.2 and {{ measurement_confidence }} > 0.7 then 'ACCEPTABLE_PLACEMENT'
        else 'POOR_PLACEMENT'
    end
{% endmacro %}

{% macro calculate_sensor_correlation_score(sensor1_value, sensor2_value, expected_correlation) %}
    -- Calculate correlation between two sensors
    case 
        when {{ sensor1_value }} is not null and {{ sensor2_value }} is not null then
            case 
                when abs({{ sensor1_value }} - {{ sensor2_value }}) <= {{ expected_correlation }} then 100
                when abs({{ sensor1_value }} - {{ sensor2_value }}) <= {{ expected_correlation }} * 2 then 80
                when abs({{ sensor1_value }} - {{ sensor2_value }}) <= {{ expected_correlation }} * 3 then 60
                else 40
            end
        else null
    end
{% endmacro %}

{% macro get_sensor_network_health_score(sensor_count, active_sensors, avg_performance_score) %}
    -- Calculate overall sensor network health score
    case 
        when {{ sensor_count }} > 0 then
            (
                ({{ active_sensors }}::float / {{ sensor_count }}) * 50 +
                {{ avg_performance_score }} * 0.5
            )
        else 0
    end
{% endmacro %}
