# Snowflake Schema Definitions

This directory contains modular Snowflake schema definitions for the PBF-LB/M manufacturing data pipeline. Each schema file defines the column names, data types, and constraints for Snowflake tables organized by data source.

## Directory Structure

```
snowflake_schemas/
├── __init__.py                 # Module initialization and exports
├── postgresql_schemas.py       # PostgreSQL table schemas
├── mongodb_schemas.py          # MongoDB collection schemas
├── cassandra_schemas.py        # Cassandra table schemas
├── redis_schemas.py            # Redis cache schemas
├── elasticsearch_schemas.py    # Elasticsearch index schemas
├── neo4j_schemas.py           # Neo4j graph schemas
├── schema_factory.py          # Schema factory and utilities
└── README.md                  # This file
```

## Schema Organization

### Data Sources
- **PostgreSQL**: 6 tables (pbf_process_data, powder_bed_data, ct_scan_data, ispm_monitoring_data, powder_bed_defects, ct_scan_defect_types)
- **MongoDB**: 8 collections (process_images, ct_scan_images, powder_bed_images, machine_build_files, model_3d_files, raw_sensor_data, process_logs, machine_configurations)
- **Cassandra**: 4 tables (sensor_readings, machine_events, process_monitoring, alert_events)
- **Redis**: 6 cache types (process_cache, machine_status_cache, sensor_readings_cache, analytics_cache, job_queue_items, user_sessions)
- **Elasticsearch**: 7 indices (pbf_process, sensor_readings, quality_metrics, machine_status, build_instructions, analytics, search_logs)
- **Neo4j**: 2 types (nodes, relationships)

### Data Types
| Snowflake Type | Description | Usage |
|----------------|-------------|-------|
| `VARCHAR(n)` | Variable character string | Text data, IDs, names |
| `TEXT` | Large text | Long text content |
| `NUMBER(p,s)` | Numeric data | Measurements, counts, scores |
| `BOOLEAN` | True/false values | Flags, status indicators |
| `TIMESTAMP` | Date and time | Timestamps, created_at, updated_at |
| `VARIANT` | JSON data | Complex objects, metadata |

## Usage Examples

### Basic Schema Access
```python
from src.data_pipeline.processing.schema.snowflake_schemas import schema_factory

# Get schema for PostgreSQL pbf_process_data table
schema = schema_factory.get_schema('postgresql', 'pbf_process_data')

# Get all schemas for MongoDB
mongodb_schemas = schema_factory.get_all_schemas_for_source('mongodb')
```

### Schema Validation
```python
# Validate data against schema
data = {
    'PROCESS_ID': 'proc_123',
    'MACHINE_ID': 'machine_456',
    'PROCESS_PARAMETERS': {'power': 100, 'speed': 50}
}

result = schema_factory.validate_schema('postgresql', 'pbf_process_data', data)
print(f"Valid: {result['valid']}")
print(f"Errors: {result['errors']}")
print(f"Warnings: {result['warnings']}")
```

### SQL Generation
```python
# Generate CREATE TABLE SQL
create_sql = schema_factory.get_create_table_sql('postgresql', 'pbf_process_data', 'RAW')
print(create_sql)

# Generate INSERT SQL template
insert_sql = schema_factory.get_insert_sql('mongodb', 'process_images', 'RAW')
print(insert_sql)
```

### Schema Summary
```python
# Get summary of all schemas
summary = schema_factory.get_schema_summary()
print(f"Total sources: {summary['total_sources']}")
print(f"Total tables: {summary['total_tables']}")

for source, info in summary['sources'].items():
    print(f"{source}: {info['table_count']} tables")
```

## Schema Design Principles

### 1. Modularity
- Each data source has its own schema file
- Easy to extend with new sources
- Clear separation of concerns

### 2. Consistency
- Standardized naming conventions
- Consistent data type usage
- Uniform metadata handling

### 3. Extensibility
- Easy to add new tables/collections
- Simple to modify existing schemas
- Support for schema evolution

### 4. Validation
- Built-in data validation
- Type checking and constraints
- Error reporting and warnings

## Adding New Schemas

### 1. Create Schema File
```python
# new_source_schemas.py
NEW_SOURCE_TABLE_SCHEMA = {
    "ID": "VARCHAR(255) NOT NULL",
    "NAME": "VARCHAR(255)",
    "DATA": "VARIANT",
    "CREATED_AT": "TIMESTAMP"
}

NEW_SOURCE_SCHEMAS = {
    "table_name": NEW_SOURCE_TABLE_SCHEMA,
}
```

### 2. Update Factory
```python
# schema_factory.py
from .new_source_schemas import NEW_SOURCE_SCHEMAS

class SnowflakeSchemaFactory:
    def __init__(self):
        self.schemas = {
            # ... existing schemas ...
            'new_source': NEW_SOURCE_SCHEMAS,
        }
```

### 3. Update Exports
```python
# __init__.py
from .new_source_schemas import *
from .schema_factory import SnowflakeSchemaFactory

__all__ = [
    # ... existing exports ...
    'NEW_SOURCE_TABLE_SCHEMA',
    'NEW_SOURCE_SCHEMAS',
]
```

## Best Practices

### 1. Naming Conventions
- Use UPPER_CASE for column names
- Prefix with source name for table names
- Use descriptive, consistent naming

### 2. Data Types
- Use VARIANT for JSON/complex data
- Use appropriate VARCHAR lengths
- Use NUMBER with precision and scale
- Use TIMESTAMP for date/time data

### 3. Constraints
- Add NOT NULL for required fields
- Use appropriate data types
- Consider indexing requirements

### 4. Documentation
- Document schema changes
- Include usage examples
- Maintain version history

## Troubleshooting

### Common Issues

#### Schema Not Found
```
Error: Schema not found for postgresql.invalid_table
```
**Solution**: Check table name and source name spelling

#### Data Type Mismatch
```
Error: Expression type does not match column data type
```
**Solution**: Ensure data matches schema data types

#### Validation Errors
```
Error: Required field 'PROCESS_ID' is missing
```
**Solution**: Check data structure and required fields

### Debug Commands
```python
# List all available sources
sources = schema_factory.list_sources()
print(sources)

# List tables for a source
tables = schema_factory.list_tables_for_source('postgresql')
print(tables)

# Get schema summary
summary = schema_factory.get_schema_summary()
print(summary)
```

## Future Enhancements

### Planned Features
- **Schema Versioning**: Track schema changes over time
- **Migration Scripts**: Automate schema updates
- **Validation Rules**: Custom validation logic
- **Performance Optimization**: Schema optimization recommendations

### Integration Points
- **dbt**: Data transformation and modeling
- **Apache Airflow**: Schema management workflows
- **Data Quality**: Automated data quality checks
- **Monitoring**: Schema usage and performance metrics
