"""
Snowflake Schema Factory

This module provides a factory pattern for managing and retrieving Snowflake schemas.
It centralizes schema management and provides utilities for schema operations.
"""

from typing import Dict, Any, Optional, List
from .postgresql_schemas import POSTGRESQL_SCHEMAS
from .mongodb_schemas import MONGODB_SCHEMAS
from .cassandra_schemas import CASSANDRA_SCHEMAS
from .redis_schemas import REDIS_SCHEMAS
from .elasticsearch_schemas import ELASTICSEARCH_SCHEMAS
from .neo4j_schemas import NEO4J_SCHEMAS


class SnowflakeSchemaFactory:
    """
    Factory class for managing Snowflake schemas across all data sources.
    Provides centralized access to schema definitions and utilities.
    """
    
    def __init__(self):
        """Initialize the schema factory with all available schemas."""
        self.schemas = {
            'postgresql': POSTGRESQL_SCHEMAS,
            'mongodb': MONGODB_SCHEMAS,
            'cassandra': CASSANDRA_SCHEMAS,
            'redis': REDIS_SCHEMAS,
            'elasticsearch': ELASTICSEARCH_SCHEMAS,
            'neo4j': NEO4J_SCHEMAS,
        }
    
    def get_schema(self, source: str, table: str) -> Optional[Dict[str, str]]:
        """
        Get schema for a specific source and table.
        
        Args:
            source: Data source name (postgresql, mongodb, etc.)
            table: Table/collection name
            
        Returns:
            Schema dictionary or None if not found
        """
        source_schemas = self.schemas.get(source.lower())
        if not source_schemas:
            return None
        return source_schemas.get(table.lower())
    
    def get_all_schemas_for_source(self, source: str) -> Dict[str, Dict[str, str]]:
        """
        Get all schemas for a specific data source.
        
        Args:
            source: Data source name
            
        Returns:
            Dictionary of all schemas for the source
        """
        return self.schemas.get(source.lower(), {})
    
    def get_all_schemas(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """
        Get all schemas for all data sources.
        
        Returns:
            Nested dictionary of all schemas
        """
        return self.schemas
    
    def list_sources(self) -> List[str]:
        """
        List all available data sources.
        
        Returns:
            List of source names
        """
        return list(self.schemas.keys())
    
    def list_tables_for_source(self, source: str) -> List[str]:
        """
        List all tables for a specific data source.
        
        Args:
            source: Data source name
            
        Returns:
            List of table names
        """
        source_schemas = self.schemas.get(source.lower(), {})
        return list(source_schemas.keys())
    
    def get_table_name(self, source: str, table: str) -> str:
        """
        Get the Snowflake table name for a source and table.
        
        Args:
            source: Data source name
            table: Table/collection name
            
        Returns:
            Snowflake table name (e.g., POSTGRESQL_PBF_PROCESS_DATA)
        """
        return f"{source.upper()}_{table.upper()}"
    
    def validate_schema(self, source: str, table: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data against schema.
        
        Args:
            source: Data source name
            table: Table/collection name
            data: Data to validate
            
        Returns:
            Validation result with errors and warnings
        """
        schema = self.get_schema(source, table)
        if not schema:
            return {
                'valid': False,
                'errors': [f"Schema not found for {source}.{table}"],
                'warnings': []
            }
        
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = [col for col, defn in schema.items() if 'NOT NULL' in defn]
        for field in required_fields:
            if field not in data:
                errors.append(f"Required field '{field}' is missing")
        
        # Check data types (basic validation)
        for field, value in data.items():
            if field in schema:
                expected_type = schema[field]
                if 'VARIANT' in expected_type and not isinstance(value, (dict, list, str)):
                    warnings.append(f"Field '{field}' should be VARIANT but got {type(value).__name__}")
                elif 'VARCHAR' in expected_type and not isinstance(value, str):
                    warnings.append(f"Field '{field}' should be VARCHAR but got {type(value).__name__}")
                elif 'NUMBER' in expected_type and not isinstance(value, (int, float)):
                    warnings.append(f"Field '{field}' should be NUMBER but got {type(value).__name__}")
                elif 'BOOLEAN' in expected_type and not isinstance(value, bool):
                    warnings.append(f"Field '{field}' should be BOOLEAN but got {type(value).__name__}")
                elif 'TIMESTAMP' in expected_type and not isinstance(value, (str, int, float)):
                    warnings.append(f"Field '{field}' should be TIMESTAMP but got {type(value).__name__}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def get_create_table_sql(self, source: str, table: str, schema_name: str = "RAW") -> str:
        """
        Generate CREATE TABLE SQL for a schema.
        
        Args:
            source: Data source name
            table: Table/collection name
            schema_name: Snowflake schema name
            
        Returns:
            CREATE TABLE SQL statement
        """
        schema = self.get_schema(source, table)
        if not schema:
            raise ValueError(f"Schema not found for {source}.{table}")
        
        table_name = self.get_table_name(source, table)
        qualified_table = f"{schema_name}.{table_name}"
        
        columns = []
        for col_name, col_def in schema.items():
            columns.append(f"    {col_name} {col_def}")
        
        columns_str = ',\n'.join(columns)
        sql = f"""CREATE TABLE IF NOT EXISTS {qualified_table} (
{columns_str}
);"""
        
        return sql
    
    def get_insert_sql(self, source: str, table: str, schema_name: str = "RAW") -> str:
        """
        Generate INSERT SQL template for a schema.
        
        Args:
            source: Data source name
            table: Table/collection name
            schema_name: Snowflake schema name
            
        Returns:
            INSERT SQL template
        """
        schema = self.get_schema(source, table)
        if not schema:
            raise ValueError(f"Schema not found for {source}.{table}")
        
        table_name = self.get_table_name(source, table)
        qualified_table = f"{schema_name}.{table_name}"
        
        columns = list(schema.keys())
        placeholders = [f":{col}" for col in columns]
        
        sql = f"""INSERT INTO {qualified_table} ({', '.join(columns)})
VALUES ({', '.join(placeholders)});"""
        
        return sql
    
    def get_schema_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all available schemas.
        
        Returns:
            Summary dictionary with counts and details
        """
        summary = {
            'total_sources': len(self.schemas),
            'total_tables': sum(len(tables) for tables in self.schemas.values()),
            'sources': {}
        }
        
        for source, tables in self.schemas.items():
            summary['sources'][source] = {
                'table_count': len(tables),
                'tables': list(tables.keys())
            }
        
        return summary


# Global schema factory instance
schema_factory = SnowflakeSchemaFactory()
