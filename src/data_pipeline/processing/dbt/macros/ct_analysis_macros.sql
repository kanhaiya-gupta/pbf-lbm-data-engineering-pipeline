-- CT Analysis Macros
-- Reusable SQL logic for CT scan data analysis and quality assessment

{% macro calculate_ct_quality_score(contrast_to_noise_ratio, signal_to_noise_ratio, spatial_resolution, uniformity, artifact_penalty) %}
    -- Calculate overall CT scan quality score
    (
        case when {{ contrast_to_noise_ratio }} is not null then 
            least(100, ({{ contrast_to_noise_ratio }} / 10.0) * 100) else 0 end +
        case when {{ signal_to_noise_ratio }} is not null then 
            least(100, ({{ signal_to_noise_ratio }} / 20.0) * 100) else 0 end +
        case when {{ spatial_resolution }} is not null then 
            least(100, ({{ spatial_resolution }} / 5.0) * 100) else 0 end +
        case when {{ uniformity }} is not null then {{ uniformity }} else 0 end -
        coalesce({{ artifact_penalty }}, 0)
    ) / 4.0
{% endmacro %}

{% macro get_ct_quality_grade(quality_score) %}
    -- Get CT quality grade based on quality score
    case 
        when {{ quality_score }} >= 90 then 'EXCELLENT'
        when {{ quality_score }} >= 80 then 'GOOD'
        when {{ quality_score }} >= 70 then 'ACCEPTABLE'
        when {{ quality_score }} >= 60 then 'POOR'
        else 'UNACCEPTABLE'
    end
{% endmacro %}

{% macro calculate_artifact_penalty(artifacts_detected, artifact_severity) %}
    -- Calculate artifact penalty score
    case 
        when {{ artifacts_detected }} = true then
            case 
                when {{ artifact_severity }} = 'NONE' then 0
                when {{ artifact_severity }} = 'MINIMAL' then 5
                when {{ artifact_severity }} = 'MODERATE' then 15
                when {{ artifact_severity }} = 'SEVERE' then 30
                else 0
            end
        else 0
    end
{% endmacro %}

{% macro calculate_scan_efficiency(number_of_projections, exposure_time, scan_duration) %}
    -- Calculate scan efficiency
    case 
        when {{ scan_duration }} > 0 then
            least(1.0, ({{ number_of_projections }} * {{ exposure_time }} / 60.0) / {{ scan_duration }})
        else 0.0
    end
{% endmacro %}

{% macro get_voxel_volume_mm3(voxel_size) %}
    -- Calculate voxel volume in cubic millimeters
    power({{ voxel_size }} / 1000.0, 3)
{% endmacro %}

{% macro get_total_volume_mm3(physical_width, physical_height, physical_depth) %}
    -- Calculate total scanned volume in cubic millimeters
    {{ physical_width }} * {{ physical_height }} * {{ physical_depth }}
{% endmacro %}

{% macro get_file_size_mb(file_size) %}
    -- Convert file size to megabytes
    {{ file_size }} / (1024.0 * 1024.0)
{% endmacro %}

{% macro calculate_defect_density(total_defects, total_volume) %}
    -- Calculate defect density per cubic millimeter
    case 
        when {{ total_volume }} > 0 then
            {{ total_defects }} / {{ total_volume }}
        else null
    end
{% endmacro %}

{% macro get_defect_severity_classification(total_defects) %}
    -- Classify defect severity based on total defects
    case 
        when {{ total_defects }} = 0 then 'NO_DEFECTS'
        when {{ total_defects }} between 1 and 5 then 'LOW_DEFECTS'
        when {{ total_defects }} between 6 and 15 then 'MODERATE_DEFECTS'
        when {{ total_defects }} between 16 and 30 then 'HIGH_DEFECTS'
        when {{ total_defects }} > 30 then 'CRITICAL_DEFECTS'
        else 'UNKNOWN_DEFECTS'
    end
{% endmacro %}

{% macro get_acceptance_recommendation(quality_score, total_defects, artifacts_detected, artifact_severity) %}
    -- Get acceptance recommendation based on CT analysis
    case 
        when {{ quality_score }} >= 90 and {{ total_defects }} = 0 and 
             ({{ artifacts_detected }} = false or {{ artifact_severity }} = 'NONE') then 'ACCEPT'
        when {{ quality_score }} >= 80 and {{ total_defects }} <= 5 and 
             ({{ artifacts_detected }} = false or {{ artifact_severity }} in ('NONE', 'MINIMAL')) then 'ACCEPT'
        when {{ quality_score }} >= 70 and {{ total_defects }} <= 10 and 
             {{ artifact_severity }} in ('NONE', 'MINIMAL', 'MODERATE') then 'CONDITIONAL_ACCEPT'
        when {{ quality_score }} >= 60 and {{ total_defects }} <= 20 then 'REQUIRES_REVIEW'
        else 'REJECT'
    end
{% endmacro %}

{% macro calculate_image_resolution_score(image_width, image_height, image_depth, physical_width, physical_height, physical_depth) %}
    -- Calculate image resolution score
    case 
        when {{ physical_width }} > 0 and {{ physical_height }} > 0 and {{ physical_depth }} > 0 then
            (
                ({{ image_width }} / {{ physical_width }}) * 0.4 +
                ({{ image_height }} / {{ physical_height }}) * 0.4 +
                ({{ image_depth }} / {{ physical_depth }}) * 0.2
            )
        else null
    end
{% endmacro %}

{% macro get_scan_parameter_optimization_score(voltage, current, exposure_time, number_of_projections) %}
    -- Calculate scan parameter optimization score
    case 
        when {{ voltage }} between 80 and 200 and {{ current }} between 50 and 300 and 
             {{ exposure_time }} between 0.1 and 2.0 and {{ number_of_projections }} between 500 and 2000 then 100
        when {{ voltage }} between 60 and 250 and {{ current }} between 30 and 400 and 
             {{ exposure_time }} between 0.05 and 5.0 and {{ number_of_projections }} between 300 and 3000 then 80
        when {{ voltage }} between 40 and 300 and {{ current }} between 20 and 500 and 
             {{ exposure_time }} between 0.01 and 10.0 and {{ number_of_projections }} between 200 and 5000 then 60
        else 40
    end
{% endmacro %}

{% macro calculate_scan_duration_efficiency(expected_duration, actual_duration) %}
    -- Calculate scan duration efficiency
    case 
        when {{ actual_duration }} > 0 then
            least(1.0, {{ expected_duration }} / {{ actual_duration }})
        else 0.0
    end
{% endmacro %}

{% macro get_compression_efficiency(file_size, voxel_count) %}
    -- Calculate compression efficiency
    case 
        when {{ voxel_count }} > 0 then
            {{ file_size }} / {{ voxel_count }}
        else null
    end
{% endmacro %}

{% macro calculate_contrast_resolution_score(contrast_to_noise_ratio, signal_to_noise_ratio) %}
    -- Calculate contrast resolution score
    case 
        when {{ contrast_to_noise_ratio }} is not null and {{ signal_to_noise_ratio }} is not null then
            ({{ contrast_to_noise_ratio }} + {{ signal_to_noise_ratio }}) / 2.0
        when {{ contrast_to_noise_ratio }} is not null then
            {{ contrast_to_noise_ratio }}
        when {{ signal_to_noise_ratio }} is not null then
            {{ signal_to_noise_ratio }}
        else null
    end
{% endmacro %}

{% macro get_scan_quality_assessment(quality_score, defect_count, artifact_severity) %}
    -- Get comprehensive scan quality assessment
    case 
        when {{ quality_score }} >= 90 and {{ defect_count }} = 0 and {{ artifact_severity }} = 'NONE' then 'PREMIUM_QUALITY'
        when {{ quality_score }} >= 80 and {{ defect_count }} <= 5 and {{ artifact_severity }} in ('NONE', 'MINIMAL') then 'HIGH_QUALITY'
        when {{ quality_score }} >= 70 and {{ defect_count }} <= 10 and {{ artifact_severity }} in ('NONE', 'MINIMAL', 'MODERATE') then 'GOOD_QUALITY'
        when {{ quality_score }} >= 60 and {{ defect_count }} <= 20 then 'ACCEPTABLE_QUALITY'
        when {{ quality_score }} >= 50 and {{ defect_count }} <= 50 then 'POOR_QUALITY'
        else 'UNACCEPTABLE_QUALITY'
    end
{% endmacro %}

{% macro calculate_scan_cost_estimate(scan_duration, voltage, current, cost_per_kwh) %}
    -- Calculate estimated scan cost
    case 
        when {{ scan_duration }} > 0 and {{ voltage }} > 0 and {{ current }} > 0 then
            ({{ voltage }} * {{ current }} * {{ scan_duration }} / 60.0 / 1000.0) * coalesce({{ cost_per_kwh }}, 0.12)
        else null
    end
{% endmacro %}

{% macro get_scan_recommendation(quality_score, defect_count, scan_efficiency, cost_estimate) %}
    -- Get scan recommendation based on multiple factors
    case 
        when {{ quality_score }} >= 90 and {{ defect_count }} = 0 and {{ scan_efficiency }} >= 0.8 then 'OPTIMAL_SCAN'
        when {{ quality_score }} >= 80 and {{ defect_count }} <= 5 and {{ scan_efficiency }} >= 0.7 then 'GOOD_SCAN'
        when {{ quality_score }} >= 70 and {{ defect_count }} <= 10 and {{ scan_efficiency }} >= 0.6 then 'ACCEPTABLE_SCAN'
        when {{ quality_score }} >= 60 and {{ defect_count }} <= 20 then 'POOR_SCAN'
        else 'FAILED_SCAN'
    end
{% endmacro %}

{% macro calculate_scan_repeatability_score(quality_score_stddev, defect_count_stddev) %}
    -- Calculate scan repeatability score
    case 
        when {{ quality_score_stddev }} is not null and {{ defect_count_stddev }} is not null then
            greatest(0, 100 - ({{ quality_score_stddev }} * 2 + {{ defect_count_stddev }} * 5))
        when {{ quality_score_stddev }} is not null then
            greatest(0, 100 - {{ quality_score_stddev }} * 2)
        when {{ defect_count_stddev }} is not null then
            greatest(0, 100 - {{ defect_count_stddev }} * 5)
        else null
    end
{% endmacro %}

{% macro get_scan_trend_analysis(quality_scores, defect_counts) %}
    -- Analyze scan trend over time
    case 
        when count(*) >= 3 then
            case 
                when avg({{ quality_scores }}) > lag(avg({{ quality_scores }}), 1) over (order by created_at) and
                     avg({{ defect_counts }}) < lag(avg({{ defect_counts }}), 1) over (order by created_at) then 'IMPROVING'
                when avg({{ quality_scores }}) < lag(avg({{ quality_scores }}), 1) over (order by created_at) and
                     avg({{ defect_counts }}) > lag(avg({{ defect_counts }}), 1) over (order by created_at) then 'DEGRADING'
                else 'STABLE'
            end
        else 'INSUFFICIENT_DATA'
    end
{% endmacro %}

{% macro calculate_scan_throughput(scan_duration, total_volume) %}
    -- Calculate scan throughput (volume per hour)
    case 
        when {{ scan_duration }} > 0 then
            {{ total_volume }} / {{ scan_duration }} * 60.0
        else null
    end
{% endmacro %}

{% macro get_scan_optimization_priority(quality_score, scan_efficiency, cost_estimate, defect_count) %}
    -- Get scan optimization priority
    case 
        when {{ quality_score }} < 60 or {{ defect_count }} > 50 then 'CRITICAL_OPTIMIZATION'
        when {{ quality_score }} < 70 or {{ scan_efficiency }} < 0.5 then 'HIGH_OPTIMIZATION'
        when {{ quality_score }} < 80 or {{ scan_efficiency }} < 0.7 then 'MEDIUM_OPTIMIZATION'
        when {{ quality_score }} < 90 or {{ scan_efficiency }} < 0.8 then 'LOW_OPTIMIZATION'
        else 'MAINTAIN_CURRENT'
    end
{% endmacro %}
