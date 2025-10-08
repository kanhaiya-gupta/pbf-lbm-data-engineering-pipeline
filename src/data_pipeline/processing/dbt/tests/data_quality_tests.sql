-- Data Quality Tests
-- Comprehensive data quality tests for all PBF-LB/M data pipeline models

-- Test 1: Check for null values in critical fields
select 
    'stg_pbf_process_data' as model_name,
    'process_id' as column_name,
    count(*) as null_count
from {{ ref('stg_pbf_process_data') }}
where process_id is null

union all

select 
    'stg_pbf_process_data' as model_name,
    'machine_id' as column_name,
    count(*) as null_count
from {{ ref('stg_pbf_process_data') }}
where machine_id is null

union all

select 
    'stg_pbf_process_data' as model_name,
    'timestamp' as column_name,
    count(*) as null_count
from {{ ref('stg_pbf_process_data') }}
where timestamp is null

union all

select 
    'stg_pbf_process_data' as model_name,
    'temperature' as column_name,
    count(*) as null_count
from {{ ref('stg_pbf_process_data') }}
where temperature is null

union all

select 
    'stg_pbf_process_data' as model_name,
    'pressure' as column_name,
    count(*) as null_count
from {{ ref('stg_pbf_process_data') }}
where pressure is null

union all

select 
    'stg_pbf_process_data' as model_name,
    'laser_power' as column_name,
    count(*) as null_count
from {{ ref('stg_pbf_process_data') }}
where laser_power is null

union all

select 
    'stg_pbf_process_data' as model_name,
    'scan_speed' as column_name,
    count(*) as null_count
from {{ ref('stg_pbf_process_data') }}
where scan_speed is null

union all

select 
    'stg_pbf_process_data' as model_name,
    'layer_height' as column_name,
    count(*) as null_count
from {{ ref('stg_pbf_process_data') }}
where layer_height is null

union all

-- ISPM Monitoring Data Tests
select 
    'stg_ispm_monitoring_data' as model_name,
    'monitoring_id' as column_name,
    count(*) as null_count
from {{ ref('stg_ispm_monitoring_data') }}
where monitoring_id is null

union all

select 
    'stg_ispm_monitoring_data' as model_name,
    'process_id' as column_name,
    count(*) as null_count
from {{ ref('stg_ispm_monitoring_data') }}
where process_id is null

union all

select 
    'stg_ispm_monitoring_data' as model_name,
    'sensor_id' as column_name,
    count(*) as null_count
from {{ ref('stg_ispm_monitoring_data') }}
where sensor_id is null

union all

select 
    'stg_ispm_monitoring_data' as model_name,
    'timestamp' as column_name,
    count(*) as null_count
from {{ ref('stg_ispm_monitoring_data') }}
where timestamp is null

union all

select 
    'stg_ispm_monitoring_data' as model_name,
    'sensor_type' as column_name,
    count(*) as null_count
from {{ ref('stg_ispm_monitoring_data') }}
where sensor_type is null

union all

select 
    'stg_ispm_monitoring_data' as model_name,
    'measurement_value' as column_name,
    count(*) as null_count
from {{ ref('stg_ispm_monitoring_data') }}
where measurement_value is null

union all

select 
    'stg_ispm_monitoring_data' as model_name,
    'unit' as column_name,
    count(*) as null_count
from {{ ref('stg_ispm_monitoring_data') }}
where unit is null

union all

-- CT Scan Data Tests
select 
    'stg_ct_scan_data' as model_name,
    'scan_id' as column_name,
    count(*) as null_count
from {{ ref('stg_ct_scan_data') }}
where scan_id is null

union all

select 
    'stg_ct_scan_data' as model_name,
    'process_id' as column_name,
    count(*) as null_count
from {{ ref('stg_ct_scan_data') }}
where process_id is null

union all

select 
    'stg_ct_scan_data' as model_name,
    'scan_type' as column_name,
    count(*) as null_count
from {{ ref('stg_ct_scan_data') }}
where scan_type is null

union all

select 
    'stg_ct_scan_data' as model_name,
    'processing_status' as column_name,
    count(*) as null_count
from {{ ref('stg_ct_scan_data') }}
where processing_status is null

union all

select 
    'stg_ct_scan_data' as model_name,
    'voltage' as column_name,
    count(*) as null_count
from {{ ref('stg_ct_scan_data') }}
where voltage is null

union all

select 
    'stg_ct_scan_data' as model_name,
    'current' as column_name,
    count(*) as null_count
from {{ ref('stg_ct_scan_data') }}
where current is null

union all

select 
    'stg_ct_scan_data' as model_name,
    'exposure_time' as column_name,
    count(*) as null_count
from {{ ref('stg_ct_scan_data') }}
where exposure_time is null

union all

select 
    'stg_ct_scan_data' as model_name,
    'number_of_projections' as column_name,
    count(*) as null_count
from {{ ref('stg_ct_scan_data') }}
where number_of_projections is null

union all

select 
    'stg_ct_scan_data' as model_name,
    'voxel_size' as column_name,
    count(*) as null_count
from {{ ref('stg_ct_scan_data') }}
where voxel_size is null

union all

select 
    'stg_ct_scan_data' as model_name,
    'file_size' as column_name,
    count(*) as null_count
from {{ ref('stg_ct_scan_data') }}
where file_size is null

union all

-- Powder Bed Data Tests
select 
    'stg_powder_bed_data' as model_name,
    'bed_id' as column_name,
    count(*) as null_count
from {{ ref('stg_powder_bed_data') }}
where bed_id is null

union all

select 
    'stg_powder_bed_data' as model_name,
    'process_id' as column_name,
    count(*) as null_count
from {{ ref('stg_powder_bed_data') }}
where process_id is null

union all

select 
    'stg_powder_bed_data' as model_name,
    'layer_number' as column_name,
    count(*) as null_count
from {{ ref('stg_powder_bed_data') }}
where layer_number is null

union all

select 
    'stg_powder_bed_data' as model_name,
    'timestamp' as column_name,
    count(*) as null_count
from {{ ref('stg_powder_bed_data') }}
where timestamp is null

union all

select 
    'stg_powder_bed_data' as model_name,
    'material_type' as column_name,
    count(*) as null_count
from {{ ref('stg_powder_bed_data') }}
where material_type is null

union all

select 
    'stg_powder_bed_data' as model_name,
    'uniformity_score' as column_name,
    count(*) as null_count
from {{ ref('stg_powder_bed_data') }}
where uniformity_score is null

union all

select 
    'stg_powder_bed_data' as model_name,
    'coverage_percentage' as column_name,
    count(*) as null_count
from {{ ref('stg_powder_bed_data') }}
where coverage_percentage is null

union all

select 
    'stg_powder_bed_data' as model_name,
    'thickness_consistency' as column_name,
    count(*) as null_count
from {{ ref('stg_powder_bed_data') }}
where thickness_consistency is null

union all

select 
    'stg_powder_bed_data' as model_name,
    'processing_status' as column_name,
    count(*) as null_count
from {{ ref('stg_powder_bed_data') }}
where processing_status is null
