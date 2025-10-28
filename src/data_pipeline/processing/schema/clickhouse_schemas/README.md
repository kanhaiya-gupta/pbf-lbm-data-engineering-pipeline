# ClickHouse Schemas for PBF-LB/M Data Warehouse

This directory contains comprehensive ClickHouse schema definitions for the PBF-LB/M data warehouse, optimized for analytics and time-series queries.

## Schema Files

**Total: 18 Comprehensive ClickHouse Schemas**

### Core Operational Data
- **`pbf_processes.sql`** - PBF manufacturing process data with quality metrics
- **`machine_status.sql`** - Real-time machine health and operational status
- **`sensor_readings.sql`** - Time-series sensor data from manufacturing processes
- **`analytics.sql`** - Business intelligence and predictive analytics data

### MongoDB Integration Data
- **`process_logs.sql`** - Process logs and annotations from MongoDB
- **`machine_configurations.sql`** - Machine configuration and calibration data
- **`raw_sensor_data.sql`** - Raw sensor data files and metadata
- **`3d_model_files.sql`** - 3D model files and geometry metadata
- **`build_instructions.sql`** - Build instructions and process documentation
- **`ct_scan_images.sql`** - CT scan images and defect analysis data
- **`powder_bed_images.sql`** - Powder bed images and powder analysis data
- **`process_images.sql`** - Process monitoring images and quality analysis
- **`machine_build_files.sql`** - Machine build files and configuration data

### Multi-Database Integration Data
- **`redis_cache_data.sql`** - Redis cache performance and analytics data
- **`job_queue_data.sql`** - Redis job queue processing and analytics data
- **`user_session_data.sql`** - Redis user session and authentication data
- **`cassandra_time_series.sql`** - Cassandra time-series sensor data
- **`ispm_monitoring.sql`** - ISPM industrial monitoring and anomaly detection

## Schema Features

### Performance Optimizations
- **Partitioning**: Monthly partitioning for time-series data
- **Indexing**: Bloom filters and minmax indexes for fast queries
- **Ordering**: Optimized primary keys for query performance
- **TTL**: Automatic data retention policies

### Materialized Views
Each schema includes materialized views for:
- **Real-time analytics** - Pre-computed aggregations
- **Trend analysis** - Time-series trend calculations
- **Health monitoring** - System health metrics
- **Cost analysis** - Financial performance tracking

### Data Types
- **Time-series optimized** - DateTime columns with proper indexing
- **Array support** - For tags, relationships, and multi-value fields
- **Nullable handling** - Proper null handling for optional fields
- **Precision** - Appropriate precision for financial and scientific data

## Usage

### Creating Tables
```sql
-- Execute schema files in ClickHouse
SOURCE pbf_processes.sql;
SOURCE machine_status.sql;
-- ... etc
```

### Loading Data
Use the `load_to_clickhouse.py` script to populate these tables with operational data.

### Analytics Queries
The materialized views provide pre-computed analytics:
- Quality trends over time
- Machine health monitoring
- Cost analysis by type
- Anomaly detection patterns

## Data Retention
- **Process Data**: 2 years
- **Machine Status**: 1 year  
- **Sensor Data**: 6 months
- **Logs**: 1 year
- **Configurations**: 2 years

## Performance Considerations
- All tables use MergeTree engine for optimal analytics performance
- Monthly partitioning reduces query scope
- Materialized views provide real-time aggregations
- Proper indexing ensures fast lookups

## Integration
These schemas integrate with:
- **MongoDB** - Process logs, configurations, raw sensor data
- **Elasticsearch** - Search and analytics data
- **Cassandra** - Time-series sensor data
- **Redis** - Cached operational data
- **PostgreSQL** - Relational operational data
