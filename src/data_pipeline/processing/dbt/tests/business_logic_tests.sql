-- Business Logic Tests
-- Tests for business logic validation and data consistency

-- Test 1: Validate process parameter ranges
select 
    'stg_pbf_process_data' as model_name,
    'temperature_range_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_pbf_process_data') }}
where temperature < 0 or temperature > 2000

union all

select 
    'stg_pbf_process_data' as model_name,
    'pressure_range_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_pbf_process_data') }}
where pressure < 0 or pressure > 1000

union all

select 
    'stg_pbf_process_data' as model_name,
    'laser_power_range_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_pbf_process_data') }}
where laser_power < 0 or laser_power > 1000

union all

select 
    'stg_pbf_process_data' as model_name,
    'scan_speed_range_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_pbf_process_data') }}
where scan_speed < 0 or scan_speed > 10000

union all

select 
    'stg_pbf_process_data' as model_name,
    'layer_height_range_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_pbf_process_data') }}
where layer_height < 0.01 or layer_height > 1.0

union all

-- Test 2: Validate ISPM monitoring data ranges
select 
    'stg_ispm_monitoring_data' as model_name,
    'measurement_value_range_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_ispm_monitoring_data') }}
where measurement_value < -1000 or measurement_value > 10000

union all

select 
    'stg_ispm_monitoring_data' as model_name,
    'noise_level_range_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_ispm_monitoring_data') }}
where noise_level < 0 or noise_level > 100

union all

select 
    'stg_ispm_monitoring_data' as model_name,
    'sampling_rate_range_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_ispm_monitoring_data') }}
where sampling_rate < 0 or sampling_rate > 1000000

union all

-- Test 3: Validate CT scan parameter ranges
select 
    'stg_ct_scan_data' as model_name,
    'voltage_range_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_ct_scan_data') }}
where voltage < 10 or voltage > 500

union all

select 
    'stg_ct_scan_data' as model_name,
    'current_range_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_ct_scan_data') }}
where current < 0.1 or current > 1000

union all

select 
    'stg_ct_scan_data' as model_name,
    'exposure_time_range_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_ct_scan_data') }}
where exposure_time < 0.001 or exposure_time > 60

union all

select 
    'stg_ct_scan_data' as model_name,
    'number_of_projections_range_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_ct_scan_data') }}
where number_of_projections < 100 or number_of_projections > 10000

union all

select 
    'stg_ct_scan_data' as model_name,
    'voxel_size_range_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_ct_scan_data') }}
where voxel_size < 0.1 or voxel_size > 1000

union all

select 
    'stg_ct_scan_data' as model_name,
    'file_size_range_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_ct_scan_data') }}
where file_size <= 0

union all

-- Test 4: Validate powder bed quality metrics ranges
select 
    'stg_powder_bed_data' as model_name,
    'uniformity_score_range_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_powder_bed_data') }}
where uniformity_score < 0 or uniformity_score > 100

union all

select 
    'stg_powder_bed_data' as model_name,
    'coverage_percentage_range_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_powder_bed_data') }}
where coverage_percentage < 0 or coverage_percentage > 100

union all

select 
    'stg_powder_bed_data' as model_name,
    'thickness_consistency_range_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_powder_bed_data') }}
where thickness_consistency < 0 or thickness_consistency > 100

union all

select 
    'stg_powder_bed_data' as model_name,
    'powder_density_range_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_powder_bed_data') }}
where powder_density < 0.1 or powder_density > 20

union all

select 
    'stg_powder_bed_data' as model_name,
    'flowability_range_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_powder_bed_data') }}
where flowability < 0 or flowability > 100

union all

select 
    'stg_powder_bed_data' as model_name,
    'moisture_content_range_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_powder_bed_data') }}
where moisture_content < 0 or moisture_content > 100

union all

-- Test 5: Validate data consistency between related fields
select 
    'stg_pbf_process_data' as model_name,
    'temperature_pressure_consistency' as test_name,
    count(*) as violation_count
from {{ ref('stg_pbf_process_data') }}
where temperature > 1500 and pressure > 10  -- High temperature with high pressure might indicate an issue

union all

select 
    'stg_pbf_process_data' as model_name,
    'laser_power_scan_speed_consistency' as test_name,
    count(*) as violation_count
from {{ ref('stg_pbf_process_data') }}
where laser_power > 500 and scan_speed > 5000  -- Very high power and speed combination

union all

select 
    'stg_ispm_monitoring_data' as model_name,
    'signal_quality_noise_consistency' as test_name,
    count(*) as violation_count
from {{ ref('stg_ispm_monitoring_data') }}
where signal_quality = 'EXCELLENT' and noise_level > 0.1  -- High noise with excellent signal quality is inconsistent

union all

select 
    'stg_ispm_monitoring_data' as model_name,
    'anomaly_detection_consistency' as test_name,
    count(*) as violation_count
from {{ ref('stg_ispm_monitoring_data') }}
where anomaly_detected = true and (anomaly_type is null or anomaly_severity is null)

union all

select 
    'stg_ct_scan_data' as model_name,
    'voltage_current_consistency' as test_name,
    count(*) as violation_count
from {{ ref('stg_ct_scan_data') }}
where voltage < 50 and current > 500  -- Low voltage with high current might cause issues

union all

select 
    'stg_ct_scan_data' as model_name,
    'processing_status_consistency' as test_name,
    count(*) as violation_count
from {{ ref('stg_ct_scan_data') }}
where processing_status = 'COMPLETED' and (overall_quality_score is null and total_defects is null)

union all

select 
    'stg_powder_bed_data' as model_name,
    'uniformity_coverage_consistency' as test_name,
    count(*) as violation_count
from {{ ref('stg_powder_bed_data') }}
where uniformity_score < 50 and coverage_percentage > 95  -- Low uniformity with high coverage might indicate an issue

union all

-- Test 6: Validate calculated field consistency
select 
    'stg_pbf_process_data' as model_name,
    'process_efficiency_range_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_pbf_process_data') }}
where process_efficiency < 0 or process_efficiency > 1

union all

select 
    'stg_pbf_process_data' as model_name,
    'data_completeness_range_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_pbf_process_data') }}
where data_completeness_score < 0 or data_completeness_score > 1

union all

select 
    'stg_ispm_monitoring_data' as model_name,
    'signal_quality_score_range_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_ispm_monitoring_data') }}
where signal_quality_score < 0 or signal_quality_score > 100

union all

select 
    'stg_ispm_monitoring_data' as model_name,
    'risk_score_range_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_ispm_monitoring_data') }}
where risk_score < 0 or risk_score > 100

union all

select 
    'stg_ispm_monitoring_data' as model_name,
    'measurement_confidence_range_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_ispm_monitoring_data') }}
where measurement_confidence < 0 or measurement_confidence > 1

union all

select 
    'stg_ct_scan_data' as model_name,
    'calculated_quality_score_range_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_ct_scan_data') }}
where calculated_quality_score < 0 or calculated_quality_score > 100

union all

select 
    'stg_ct_scan_data' as model_name,
    'scan_efficiency_range_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_ct_scan_data') }}
where scan_efficiency < 0 or scan_efficiency > 1

union all

select 
    'stg_powder_bed_data' as model_name,
    'calculated_quality_score_range_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_powder_bed_data') }}
where calculated_quality_score < 0 or calculated_quality_score > 100

union all

select 
    'stg_powder_bed_data' as model_name,
    'image_quality_score_range_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_powder_bed_data') }}
where image_quality_score < 0 or image_quality_score > 100

union all

select 
    'stg_powder_bed_data' as model_name,
    'moisture_risk_score_range_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_powder_bed_data') }}
where moisture_risk_score < 0 or moisture_risk_score > 100

union all

select 
    'stg_powder_bed_data' as model_name,
    'environmental_risk_score_range_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_powder_bed_data') }}
where environmental_risk_score < 0 or environmental_risk_score > 100

union all

-- Test 7: Validate enum values
select 
    'stg_pbf_process_data' as model_name,
    'atmosphere_enum_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_pbf_process_data') }}
where atmosphere is not null and atmosphere not in ('argon', 'nitrogen', 'helium', 'vacuum', 'air')

union all

select 
    'stg_pbf_process_data' as model_name,
    'quality_grade_enum_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_pbf_process_data') }}
where quality_grade is not null and quality_grade not in ('EXCELLENT', 'GOOD', 'ACCEPTABLE', 'POOR')

union all

select 
    'stg_ispm_monitoring_data' as model_name,
    'sensor_type_enum_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_ispm_monitoring_data') }}
where sensor_type not in ('THERMAL', 'OPTICAL', 'ACOUSTIC', 'VIBRATION', 'PRESSURE', 'GAS_ANALYSIS', 'MELT_POOL', 'LAYER_HEIGHT')

union all

select 
    'stg_ispm_monitoring_data' as model_name,
    'signal_quality_enum_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_ispm_monitoring_data') }}
where signal_quality is not null and signal_quality not in ('EXCELLENT', 'GOOD', 'FAIR', 'POOR', 'UNKNOWN')

union all

select 
    'stg_ispm_monitoring_data' as model_name,
    'anomaly_severity_enum_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_ispm_monitoring_data') }}
where anomaly_severity is not null and anomaly_severity not in ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')

union all

select 
    'stg_ct_scan_data' as model_name,
    'scan_type_enum_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_ct_scan_data') }}
where scan_type not in ('QUALITY_CONTROL', 'DEFECT_ANALYSIS', 'DIMENSIONAL_MEASUREMENT', 'MATERIAL_ANALYSIS', 'RESEARCH')

union all

select 
    'stg_ct_scan_data' as model_name,
    'processing_status_enum_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_ct_scan_data') }}
where processing_status not in ('PENDING', 'IN_PROGRESS', 'COMPLETED', 'FAILED', 'CANCELLED')

union all

select 
    'stg_ct_scan_data' as model_name,
    'artifact_severity_enum_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_ct_scan_data') }}
where artifact_severity is not null and artifact_severity not in ('NONE', 'MINIMAL', 'MODERATE', 'SEVERE')

union all

select 
    'stg_ct_scan_data' as model_name,
    'acceptance_status_enum_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_ct_scan_data') }}
where acceptance_status is not null and acceptance_status not in ('ACCEPTED', 'REJECTED', 'CONDITIONAL', 'REQUIRES_REVIEW')

union all

select 
    'stg_powder_bed_data' as model_name,
    'overall_quality_assessment_enum_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_powder_bed_data') }}
where overall_quality_assessment not in ('EXCELLENT', 'GOOD', 'ACCEPTABLE', 'POOR', 'UNACCEPTABLE')

union all

select 
    'stg_powder_bed_data' as model_name,
    'processing_status_enum_validation' as test_name,
    count(*) as violation_count
from {{ ref('stg_powder_bed_data') }}
where processing_status not in ('PENDING', 'IN_PROGRESS', 'COMPLETED', 'FAILED', 'CANCELLED')
