-- Powder Bed Macros
-- Reusable SQL logic for powder bed monitoring data transformations

{% macro calculate_powder_bed_quality_score(uniformity_score, coverage_percentage, thickness_consistency, surface_roughness, density_variation) %}
    -- Calculate overall powder bed quality score
    (
        {{ uniformity_score }} * 0.3 +
        {{ coverage_percentage }} * 0.3 +
        {{ thickness_consistency }} * 0.2 +
        case when {{ surface_roughness }} is not null then 
            greatest(0, 100 - {{ surface_roughness }}) * 0.1 else 0 end +
        case when {{ density_variation }} is not null then 
            greatest(0, 100 - {{ density_variation }} * 100) * 0.1 else 0 end
    )
{% endmacro %}

{% macro get_powder_bed_quality_grade(quality_score) %}
    -- Get powder bed quality grade based on quality score
    case 
        when {{ quality_score }} >= 90 then 'EXCELLENT'
        when {{ quality_score }} >= 80 then 'GOOD'
        when {{ quality_score }} >= 70 then 'ACCEPTABLE'
        when {{ quality_score }} >= 50 then 'POOR'
        else 'UNACCEPTABLE'
    end
{% endmacro %}

{% macro calculate_image_quality_score(brightness, contrast, sharpness, noise_level, red_channel, green_channel, blue_channel) %}
    -- Calculate image quality score based on image analysis
    case 
        when {{ brightness }} is not null and {{ contrast }} is not null and {{ sharpness }} is not null then
            (
                (100 - abs({{ brightness }} - 150) / 1.5) * 0.2 +
                least(100, {{ contrast }} * 2) * 0.25 +
                least(100, {{ sharpness }} * 1.3) * 0.25 +
                (100 - abs({{ red_channel }} - {{ green_channel }}) / 2.55) * 0.2
            ) - least(30, coalesce({{ noise_level }}, 0) * 10)
        else null
    end
{% endmacro %}

{% macro calculate_moisture_risk_score(moisture_content) %}
    -- Calculate moisture risk score
    case 
        when {{ moisture_content }} is null then 0
        when {{ moisture_content }} < 0.1 then 0
        when {{ moisture_content }} < 0.5 then {{ moisture_content }} * 20
        when {{ moisture_content }} < 1.0 then 10 + ({{ moisture_content }} - 0.5) * 40
        else 30 + least(70, ({{ moisture_content }} - 1.0) * 70)
    end
{% endmacro %}

{% macro calculate_environmental_risk_score(ambient_temperature, relative_humidity, vibration_level) %}
    -- Calculate environmental risk score
    (
        case when {{ ambient_temperature }} < 15 or {{ ambient_temperature }} > 30 then 
            least(30, abs({{ ambient_temperature }} - 22.5) * 3) else 0 end +
        case when {{ relative_humidity }} < 30 or {{ relative_humidity }} > 50 then 
            least(40, abs({{ relative_humidity }} - 40) * 2) else 0 end +
        case when {{ vibration_level }} is not null and {{ vibration_level }} > 0.1 then 
            least(30, {{ vibration_level }} * 100) else 0 end
    )
{% endmacro %}

{% macro get_particle_size_distribution_span(d10, d50, d90) %}
    -- Calculate particle size distribution span
    case 
        when {{ d10 }} is not null and {{ d50 }} is not null and {{ d90 }} is not null then
            ({{ d90 }} - {{ d10 }}) / {{ d50 }}
        else null
    end
{% endmacro %}

{% macro get_particle_size_category(d50) %}
    -- Categorize particle size based on d50
    case 
        when {{ d50 }} < 10 then 'FINE_POWDER'
        when {{ d50 }} between 10 and 25 then 'MEDIUM_POWDER'
        when {{ d50 }} between 25 and 50 then 'COARSE_POWDER'
        when {{ d50 }} > 50 then 'VERY_COARSE_POWDER'
        else 'UNKNOWN_PARTICLE_SIZE'
    end
{% endmacro %}

{% macro get_flowability_grade(flowability) %}
    -- Get flowability grade based on flowability score
    case 
        when {{ flowability }} >= 90 then 'EXCELLENT_FLOWABILITY'
        when {{ flowability }} >= 80 then 'GOOD_FLOWABILITY'
        when {{ flowability }} >= 70 then 'ACCEPTABLE_FLOWABILITY'
        when {{ flowability }} >= 60 then 'POOR_FLOWABILITY'
        else 'UNACCEPTABLE_FLOWABILITY'
    end
{% endmacro %}

{% macro get_defect_severity_classification(defect_count) %}
    -- Classify defect severity based on defect count
    case 
        when {{ defect_count }} = 0 then 'NO_DEFECTS'
        when {{ defect_count }} = 1 then 'SINGLE_DEFECT'
        when {{ defect_count }} between 2 and 5 then 'MODERATE_DEFECTS'
        when {{ defect_count }} between 6 and 10 then 'HIGH_DEFECTS'
        when {{ defect_count }} > 10 then 'CRITICAL_DEFECTS'
        else 'UNKNOWN_DEFECTS'
    end
{% endmacro %}

{% macro get_recommended_action(quality_grade, defect_count, moisture_risk, environmental_risk) %}
    -- Get recommended action based on powder bed data
    case 
        when {{ quality_grade }} = 'UNACCEPTABLE' then 'REJECT_LAYER'
        when {{ quality_grade }} = 'POOR' then 'INVESTIGATE_ISSUES'
        when {{ defect_count }} > 10 then 'ADDRESS_DEFECTS'
        when {{ moisture_risk }} > 50 then 'ADDRESS_MOISTURE'
        when {{ environmental_risk }} > 60 then 'ADJUST_ENVIRONMENT'
        when {{ quality_grade }} in ('EXCELLENT', 'GOOD') then 'CONTINUE_PROCESS'
        else 'MONITOR_CLOSELY'
    end
{% endmacro %}

{% macro calculate_uniformity_consistency_score(uniformity_stddev) %}
    -- Calculate uniformity consistency score
    case 
        when {{ uniformity_stddev }} is not null then
            greatest(0, 100 - {{ uniformity_stddev }} * 2)
        else null
    end
{% endmacro %}

{% macro calculate_coverage_consistency_score(coverage_stddev) %}
    -- Calculate coverage consistency score
    case 
        when {{ coverage_stddev }} is not null then
            greatest(0, 100 - {{ coverage_stddev }} * 2)
        else null
    end
{% endmacro %}

{% macro calculate_thickness_consistency_score(thickness_consistency_stddev) %}
    -- Calculate thickness consistency score
    case 
        when {{ thickness_consistency_stddev }} is not null then
            greatest(0, 100 - {{ thickness_consistency_stddev }} * 2)
        else null
    end
{% endmacro %}

{% macro get_powder_bed_stability_level(uniformity_consistency, coverage_consistency, thickness_consistency) %}
    -- Get powder bed stability level
    case 
        when {{ uniformity_consistency }} >= 90 and {{ coverage_consistency }} >= 90 and {{ thickness_consistency }} >= 90 then 'HIGHLY_STABLE'
        when {{ uniformity_consistency }} >= 80 and {{ coverage_consistency }} >= 80 and {{ thickness_consistency }} >= 80 then 'STABLE'
        when {{ uniformity_consistency }} >= 70 and {{ coverage_consistency }} >= 70 and {{ thickness_consistency }} >= 70 then 'MODERATELY_STABLE'
        when {{ uniformity_consistency }} >= 60 and {{ coverage_consistency }} >= 60 and {{ thickness_consistency }} >= 60 then 'UNSTABLE'
        else 'HIGHLY_UNSTABLE'
    end
{% endmacro %}

{% macro calculate_powder_bed_efficiency_score(uniformity_score, coverage_percentage, defect_count, processing_time) %}
    -- Calculate powder bed efficiency score
    case 
        when {{ processing_time }} > 0 then
            (
                {{ uniformity_score }} * 0.3 +
                {{ coverage_percentage }} * 0.3 +
                greatest(0, 100 - {{ defect_count }} * 10) * 0.2 +
                least(100, 100 - {{ processing_time }} * 10) * 0.2
            )
        else
            (
                {{ uniformity_score }} * 0.4 +
                {{ coverage_percentage }} * 0.4 +
                greatest(0, 100 - {{ defect_count }} * 10) * 0.2
            )
    end
{% endmacro %}

{% macro get_camera_performance_score(exposure_time, aperture, iso, image_quality_score) %}
    -- Calculate camera performance score
    case 
        when {{ exposure_time }} is not null and {{ aperture }} is not null and {{ iso }} is not null then
            (
                case when {{ exposure_time }} between 0.01 and 0.5 then 25 else 15 end +
                case when {{ aperture }} between 2.8 and 8.0 then 25 else 15 end +
                case when {{ iso }} between 100 and 800 then 25 else 15 end +
                coalesce({{ image_quality_score }}, 0) * 0.25
            )
        else coalesce({{ image_quality_score }}, 0)
    end
{% endmacro %}

{% macro get_lighting_quality_score(lighting_conditions, image_quality_score) %}
    -- Assess lighting quality based on conditions and image quality
    case 
        when {{ lighting_conditions }} = 'LED_ring_light' and {{ image_quality_score }} >= 80 then 'EXCELLENT_LIGHTING'
        when {{ lighting_conditions }} = 'LED_ring_light' and {{ image_quality_score }} >= 60 then 'GOOD_LIGHTING'
        when {{ lighting_conditions }} in ('LED_ring_light', 'LED_panel_light') and {{ image_quality_score }} >= 40 then 'ACCEPTABLE_LIGHTING'
        when {{ lighting_conditions }} in ('LED_ring_light', 'LED_panel_light', 'fluorescent_light') then 'POOR_LIGHTING'
        else 'UNKNOWN_LIGHTING'
    end
{% endmacro %}

{% macro calculate_powder_bed_health_score(uniformity_score, coverage_percentage, defect_count, moisture_risk, environmental_risk) %}
    -- Calculate overall powder bed health score
    (
        {{ uniformity_score }} * 0.25 +
        {{ coverage_percentage }} * 0.25 +
        greatest(0, 100 - {{ defect_count }} * 5) * 0.2 +
        greatest(0, 100 - {{ moisture_risk }}) * 0.15 +
        greatest(0, 100 - {{ environmental_risk }}) * 0.15
    )
{% endmacro %}

{% macro get_powder_bed_health_status(health_score) %}
    -- Get powder bed health status
    case 
        when {{ health_score }} >= 90 then 'EXCELLENT_HEALTH'
        when {{ health_score }} >= 80 then 'GOOD_HEALTH'
        when {{ health_score }} >= 70 then 'FAIR_HEALTH'
        when {{ health_score }} >= 60 then 'POOR_HEALTH'
        else 'CRITICAL_HEALTH'
    end
{% endmacro %}

{% macro calculate_powder_utilization_efficiency(powder_used, powder_available) %}
    -- Calculate powder utilization efficiency
    case 
        when {{ powder_available }} > 0 then
            ({{ powder_used }} / {{ powder_available }}) * 100
        else null
    end
{% endmacro %}

{% macro get_powder_bed_trend_analysis(uniformity_scores, coverage_percentages, defect_counts) %}
    -- Analyze powder bed trend over time
    case 
        when count(*) >= 3 then
            case 
                when avg({{ uniformity_scores }}) > lag(avg({{ uniformity_scores }}), 1) over (order by timestamp) and
                     avg({{ coverage_percentages }}) > lag(avg({{ coverage_percentages }}), 1) over (order by timestamp) and
                     avg({{ defect_counts }}) < lag(avg({{ defect_counts }}), 1) over (order by timestamp) then 'IMPROVING'
                when avg({{ uniformity_scores }}) < lag(avg({{ uniformity_scores }}), 1) over (order by timestamp) and
                     avg({{ coverage_percentages }}) < lag(avg({{ coverage_percentages }}), 1) over (order by timestamp) and
                     avg({{ defect_counts }}) > lag(avg({{ defect_counts }}), 1) over (order by timestamp) then 'DEGRADING'
                else 'STABLE'
            end
        else 'INSUFFICIENT_DATA'
    end
{% endmacro %}

{% macro get_powder_bed_optimization_priority(quality_score, defect_count, moisture_risk, environmental_risk) %}
    -- Get powder bed optimization priority
    case 
        when {{ quality_score }} < 50 or {{ defect_count }} > 20 or {{ moisture_risk }} > 80 or {{ environmental_risk }} > 80 then 'CRITICAL_OPTIMIZATION'
        when {{ quality_score }} < 70 or {{ defect_count }} > 10 or {{ moisture_risk }} > 60 or {{ environmental_risk }} > 60 then 'HIGH_OPTIMIZATION'
        when {{ quality_score }} < 80 or {{ defect_count }} > 5 or {{ moisture_risk }} > 40 or {{ environmental_risk }} > 40 then 'MEDIUM_OPTIMIZATION'
        when {{ quality_score }} < 90 or {{ defect_count }} > 2 or {{ moisture_risk }} > 20 or {{ environmental_risk }} > 20 then 'LOW_OPTIMIZATION'
        else 'MAINTAIN_CURRENT'
    end
{% endmacro %}

{% macro calculate_powder_bed_repeatability_score(uniformity_stddev, coverage_stddev, thickness_consistency_stddev) %}
    -- Calculate powder bed repeatability score
    case 
        when {{ uniformity_stddev }} is not null and {{ coverage_stddev }} is not null and {{ thickness_consistency_stddev }} is not null then
            (
                greatest(0, 100 - {{ uniformity_stddev }} * 2) +
                greatest(0, 100 - {{ coverage_stddev }} * 2) +
                greatest(0, 100 - {{ thickness_consistency_stddev }} * 2)
            ) / 3.0
        else null
    end
{% endmacro %}

{% macro get_powder_bed_certification_level(quality_score, repeatability_score, defect_rate) %}
    -- Get powder bed certification level
    case 
        when {{ quality_score }} >= 95 and {{ repeatability_score }} >= 95 and {{ defect_rate }} <= 1 then 'CERTIFIED_PREMIUM'
        when {{ quality_score }} >= 90 and {{ repeatability_score }} >= 90 and {{ defect_rate }} <= 3 then 'CERTIFIED_GOOD'
        when {{ quality_score }} >= 80 and {{ repeatability_score }} >= 80 and {{ defect_rate }} <= 5 then 'CERTIFIED_ACCEPTABLE'
        when {{ quality_score }} >= 70 and {{ repeatability_score }} >= 70 and {{ defect_rate }} <= 10 then 'CERTIFIED_MINIMUM'
        else 'NOT_CERTIFIED'
    end
{% endmacro %}
