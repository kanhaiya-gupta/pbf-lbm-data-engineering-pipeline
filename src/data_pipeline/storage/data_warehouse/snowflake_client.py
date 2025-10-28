"""
Snowflake Client for PBF-LB/M Data Pipeline

This module provides Snowflake operations for data warehouse storage.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

try:
    import snowflake.connector
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False
    snowflake = None

from src.data_pipeline.config.snowflake_config import (
    get_snowflake_config,
    get_snowflake_database_config,
    get_snowflake_warehouse_config,
)

logger = logging.getLogger(__name__)


class SnowflakeClient:
    """Snowflake client for data warehouse operations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Snowflake client.

        Args:
            config: Optional Snowflake configuration dictionary or Pydantic model
        """
        self.config = config or get_snowflake_config()
        self.connection = None
        self.cursor = None
        self.warehouse_config = None
        self.database_config = None
        self._initialize_connection()

    def _initialize_connection(self) -> None:
        """Initialize Snowflake connection."""
        if not SNOWFLAKE_AVAILABLE:
            logger.error("snowflake-connector-python not available for Snowflake operations")
            raise ImportError("snowflake-connector-python is required for Snowflake operations")

        try:
            conn_params = self._get_connection_params()
            self.connection = snowflake.connector.connect(**conn_params)
            self.cursor = self.connection.cursor()

            # Create and set database and schema
            self._setup_database_and_schema(
                conn_params["database"], conn_params["schema"]
            )
            logger.info("Snowflake connection initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Snowflake connection: {e}")
            raise

    def _get_connection_params(self) -> Dict[str, Any]:
        """Get connection parameters from configuration."""
        if hasattr(self.config, "get_connection_params"):
            return self.config.get_connection_params()
        return {
            "account": self.config.get("account", "your-account.snowflakecomputing.com"),
            "user": self.config.get("user", "your-username"),
            "password": self.config.get("password", "your-password"),
            "warehouse": self.config.get("warehouse", "PBF_WAREHOUSE"),
            "database": self.config.get("database", "PBF_ANALYTICS"),
            "schema": self.config.get("schema_name", "RAW"),
            "role": self.config.get("role", "ACCOUNTADMIN"),
            "use_ssl": self.config.get("use_ssl", True),
            "verify_ssl": self.config.get("verify_ssl", True),
        }

    def _setup_database_and_schema(self, database: str, schema: str) -> None:
        """Set up database and schema with fallback handling."""
        try:
            self.cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
            logger.info(f"Database {database} ensured to exist")
            self.cursor.execute(f"USE DATABASE {database}")
            logger.info(f"Using database {database}")
        except Exception as e:
            logger.warning(f"Could not create/use database {database}: {e}")
            try:
                self.cursor.execute("USE DATABASE SNOWFLAKE")
                logger.info("Using fallback database SNOWFLAKE")
            except Exception as fallback_e:
                logger.warning(f"Could not use fallback database: {fallback_e}")

        try:
            self.cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
            logger.info(f"Schema {schema} ensured to exist")
            self.cursor.execute(f"USE SCHEMA {schema}")
            logger.info(f"Using schema {schema}")
        except Exception as e:
            logger.warning(f"Could not create/use schema {schema}: {e}")
            try:
                self.cursor.execute("USE SCHEMA PUBLIC")
                logger.info("Using fallback schema PUBLIC")
            except Exception as fallback_e:
                logger.warning(f"Could not use fallback schema: {fallback_e}")

    # Core Database Operations
    def execute_query(
        self, query: str, params: Optional[tuple] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a SQL query and return results.

        Args:
            query: SQL query string
            params: Optional query parameters

        Returns:
            List[Dict[str, Any]]: Query results
        """
        if not self.cursor:
            logger.error("Snowflake cursor not initialized")
            return []

        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            columns = [desc[0] for desc in self.cursor.description]
            rows = self.cursor.fetchall()
            results = [dict(zip(columns, row)) for row in rows]
            logger.info(f"Executed query, returned {len(results)} rows")
            return results
        except Exception as e:
            logger.error(f"Error executing Snowflake query: {e}")
            return []

    def create_table(self, table_name: str, schema: str, if_not_exists: bool = True) -> bool:
        """
        Create a table in Snowflake.

        Args:
            table_name: Name of the table
            schema: Table schema definition
            if_not_exists: Whether to use IF NOT EXISTS clause

        Returns:
            bool: True if table creation successful, False otherwise
        """
        try:
            if_not_exists_clause = "IF NOT EXISTS" if if_not_exists else ""
            query = f"CREATE TABLE {if_not_exists_clause} {table_name} ({schema})"
            self.cursor.execute(query)
            logger.info(f"Created Snowflake table: {table_name}")
            return True
        except Exception as e:
            logger.error(f"Error creating Snowflake table {table_name}: {e}")
            return False

    def drop_table(self, table_name: str, if_exists: bool = True) -> bool:
        """
        Drop a table in Snowflake.

        Args:
            table_name: Name of the table
            if_exists: Whether to use IF EXISTS clause

        Returns:
            bool: True if table drop successful, False otherwise
        """
        try:
            if_exists_clause = "IF EXISTS" if if_exists else ""
            query = f"DROP TABLE {if_exists_clause} {table_name}"
            self.cursor.execute(query)
            logger.info(f"Dropped Snowflake table: {table_name}")
            return True
        except Exception as e:
            logger.error(f"Error dropping Snowflake table {table_name}: {e}")
            return False

    def insert_data(self, table_name: str, data: List[Dict[str, Any]]) -> bool:
        """
        Insert data into a Snowflake table.

        Args:
            table_name: Name of the table
            data: List of dictionaries containing data to insert

        Returns:
            bool: True if insert successful, False otherwise
        """
        if not data:
            logger.warning("No data to insert")
            return True

        try:
            columns = list(data[0].keys())
            columns_str = ", ".join(columns)
            placeholders = ", ".join(["%s"] * len(columns))
            query = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"
            
            # Convert problematic Python objects to JSON strings for Snowflake connector
            values_list = []
            for row in data:
                sanitized_row = []
                for col in columns:
                    value = row[col]
                    # Convert specific problematic types to JSON strings
                    if isinstance(value, dict):
                        import json
                        sanitized_row.append(json.dumps(value))
                    elif hasattr(value, '__class__') and value.__class__.__name__ == 'ObjectId':
                        sanitized_row.append(str(value))
                    elif hasattr(value, 'items') and not isinstance(value, dict):  # Neo4j Node objects
                        import json
                        sanitized_row.append(json.dumps(dict(value)))
                    elif hasattr(value, '__class__') and 'memoryview' in str(type(value)):
                        # Handle memoryview objects (PostgreSQL BYTEA)
                        sanitized_row.append(str(value))
                    elif isinstance(value, (bytes, bytearray)):
                        # Handle binary data
                        sanitized_row.append(str(value))
                    else:
                        sanitized_row.append(value)
                values_list.append(sanitized_row)
            
            self.cursor.executemany(query, values_list)
            logger.info(f"Inserted {len(data)} rows into Snowflake table: {table_name}")
            return True
        except Exception as e:
            logger.error(f"Error inserting data into Snowflake table {table_name}: {e}")
            return False

    def update_data(
        self, table_name: str, updates: Dict[str, Any], condition: str
    ) -> bool:
        """
        Update data in a Snowflake table.

        Args:
            table_name: Name of the table
            updates: Dictionary of column updates
            condition: WHERE condition for updates

        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            set_clause = ", ".join([f"{col} = %s" for col in updates.keys()])
            query = f"UPDATE {table_name} SET {set_clause} WHERE {condition}"
            self.cursor.execute(query, list(updates.values()))
            logger.info(f"Updated data in Snowflake table: {table_name}")
            return True
        except Exception as e:
            logger.error(f"Error updating data in Snowflake table {table_name}: {e}")
            return False

    def delete_data(self, table_name: str, condition: str) -> bool:
        """
        Delete data from a Snowflake table.

        Args:
            table_name: Name of the table
            condition: WHERE condition for deletion

        Returns:
            bool: True if deletion successful, False otherwise
        """
        try:
            query = f"DELETE FROM {table_name} WHERE {condition}"
            self.cursor.execute(query)
            logger.info(f"Deleted data from Snowflake table: {table_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting data from Snowflake table {table_name}: {e}")
            return False

    # Schema and Metadata Operations
    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get table schema information.

        Args:
            table_name: Name of the table

        Returns:
            List[Dict[str, Any]]: Table schema information
        """
        try:
            query = f"DESCRIBE TABLE {table_name}"
            results = self.execute_query(query)
            schema_info = [
                {
                    "column_name": row["name"],
                    "data_type": row["type"],
                    "null": row["null?"],
                    "default": row["default"],
                    "primary_key": row.get("primary key", False),
                    "unique_key": row.get("unique key", False),
                }
                for row in results
            ]
            logger.info(f"Retrieved schema for Snowflake table: {table_name}")
            return schema_info
        except Exception as e:
            logger.error(f"Error getting schema for Snowflake table {table_name}: {e}")
            return []

    def get_table_count(self, table_name: str) -> int:
        """
        Get row count for a table.

        Args:
            table_name: Name of the table

        Returns:
            int: Row count
        """
        try:
            query = f"SELECT COUNT(*) as row_count FROM {table_name}"
            results = self.execute_query(query)
            if results:
                count = results[0].get("row_count", results[0].get("COUNT", 0))
                logger.info(f"Snowflake table {table_name} has {count} rows")
                return count
            return 0
        except Exception as e:
            logger.error(f"Error getting count for Snowflake table {table_name}: {e}")
            return 0

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get comprehensive table information.

        Args:
            table_name: Name of the table

        Returns:
            Dict[str, Any]: Table information
        """
        return {
            "table_name": table_name,
            "row_count": self.get_table_count(table_name),
            "schema": self.get_table_schema(table_name),
            "timestamp": datetime.now().isoformat(),
        }

    # Configuration Setup Methods
    def setup_warehouse(self, warehouse_config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Setup Snowflake warehouse using configuration.

        Args:
            warehouse_config: Optional warehouse configuration

        Returns:
            bool: True if warehouse setup successful, False otherwise
        """
        try:
            self.warehouse_config = warehouse_config or get_snowflake_warehouse_config()
            if hasattr(self.warehouse_config, "get_create_warehouse_sql"):
                sql = self.warehouse_config.get_create_warehouse_sql()
                warehouse_name = self.warehouse_config.warehouse_name
            else:
                config = (
                    self.warehouse_config
                    if isinstance(self.warehouse_config, dict)
                    else vars(self.warehouse_config)
                )
                warehouse_name = config.get("warehouse_name", "PBF_WAREHOUSE")
                warehouse_size = config.get("warehouse_size", "SMALL")
                auto_suspend = config.get("auto_suspend", 60)
                auto_resume = config.get("auto_resume", True)
                auto_resume_clause = "TRUE" if auto_resume else "FALSE"
                sql = (
                    f"CREATE WAREHOUSE IF NOT EXISTS {warehouse_name} "
                    f"WITH WAREHOUSE_SIZE = {warehouse_size} "
                    f"AUTO_SUSPEND = {auto_suspend} "
                    f"AUTO_RESUME = {auto_resume_clause}"
                )
            self.cursor.execute(sql)
            logger.info(f"Warehouse {warehouse_name} setup completed")
            return True
        except Exception as e:
            logger.error(f"Error setting up warehouse: {e}")
            return False

    def setup_database(self, database_config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Setup Snowflake database and schemas using configuration.

        Args:
            database_config: Optional database configuration

        Returns:
            bool: True if database setup successful, False otherwise
        """
        try:
            self.database_config = database_config or get_snowflake_database_config()
            config = (
                self.database_config
                if isinstance(self.database_config, dict)
                else vars(self.database_config)
            )
            database_name = config.get("database_name", "PBF_ANALYTICS")
            schemas = config.get("schemas", ["RAW", "STAGING", "ANALYTICS", "REPORTS"])

            if hasattr(self.database_config, "get_create_database_sql"):
                db_sql = self.database_config.get_create_database_sql()
            else:
                db_sql = f"CREATE DATABASE IF NOT EXISTS {database_name}"
            self.cursor.execute(db_sql)
            logger.info(f"Database {database_name} created")

            schema_sqls = (
                self.database_config.get_create_schemas_sql()
                if hasattr(self.database_config, "get_create_schemas_sql")
                else [
                    f"CREATE SCHEMA IF NOT EXISTS {database_name}.{schema}"
                    for schema in schemas
                ]
            )
            for schema_sql in schema_sqls:
                self.cursor.execute(schema_sql)
            logger.info(f"Schemas {schemas} created in database {database_name}")
            return True
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
            return False

    def create_warehouse(
        self, warehouse_name: str, size: str = "SMALL", auto_suspend: int = 60, auto_resume: bool = True
    ) -> bool:
        """
        Create a Snowflake warehouse.

        Args:
            warehouse_name: Name of the warehouse
            size: Warehouse size (XSMALL, SMALL, MEDIUM, LARGE, XLARGE, etc.)
            auto_suspend: Auto-suspend time in seconds
            auto_resume: Whether to auto-resume

        Returns:
            bool: True if warehouse creation successful, False otherwise
        """
        try:
            auto_resume_clause = "TRUE" if auto_resume else "FALSE"
            query = (
                f"CREATE WAREHOUSE {warehouse_name} "
                f"WITH WAREHOUSE_SIZE = {size} "
                f"AUTO_SUSPEND = {auto_suspend} "
                f"AUTO_RESUME = {auto_resume_clause}"
            )
            self.cursor.execute(query)
            logger.info(f"Created Snowflake warehouse: {warehouse_name}")
            return True
        except Exception as e:
            logger.error(f"Error creating Snowflake warehouse {warehouse_name}: {e}")
            return False

    def create_database(self, database_name: str, if_not_exists: bool = True) -> bool:
        """
        Create a Snowflake database.

        Args:
            database_name: Name of the database
            if_not_exists: Whether to use IF NOT EXISTS clause

        Returns:
            bool: True if database creation successful, False otherwise
        """
        try:
            if_not_exists_clause = "IF NOT EXISTS" if if_not_exists else ""
            query = f"CREATE DATABASE {if_not_exists_clause} {database_name}"
            self.cursor.execute(query)
            logger.info(f"Created Snowflake database: {database_name}")
            return True
        except Exception as e:
            logger.error(f"Error creating Snowflake database {database_name}: {e}")
            return False

    def create_schema(
        self, schema_name: str, database_name: Optional[str] = None, if_not_exists: bool = True
    ) -> bool:
        """
        Create a Snowflake schema.

        Args:
            schema_name: Name of the schema
            database_name: Name of the database (uses current if None)
            if_not_exists: Whether to use IF NOT EXISTS clause

        Returns:
            bool: True if schema creation successful, False otherwise
        """
        try:
            if_not_exists_clause = "IF NOT EXISTS" if if_not_exists else ""
            database_clause = f"{database_name}." if database_name else ""
            query = f"CREATE SCHEMA {if_not_exists_clause} {database_clause}{schema_name}"
            self.cursor.execute(query)
            logger.info(f"Created Snowflake schema: {schema_name}")
            return True
        except Exception as e:
            logger.error(f"Error creating Snowflake schema {schema_name}: {e}")
            return False

    # Data Loading Methods
    def load_data_batch(
        self, table_name: str, data: List[Dict[str, Any]], batch_size: Optional[int] = None
    ) -> bool:
        """
        Load data in batches with enhanced error handling.

        Args:
            table_name: Name of the table
            data: List of dictionaries containing data to insert
            batch_size: Optional batch size (uses config default if None)

        Returns:
            bool: True if all batches loaded successfully, False otherwise
        """
        if not data:
            logger.warning("No data to load")
            return True

        try:
            batch_size = (
                self.config.batch_size
                if hasattr(self.config, "batch_size")
                else self.config.get("batch_size", 1000)
            ) if batch_size is None else batch_size
            total_batches = (len(data) + batch_size - 1) // batch_size
            successful_batches = 0

            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                batch_num = (i // batch_size) + 1
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} records)")
                if self.insert_data(table_name, batch):
                    successful_batches += 1
                else:
                    logger.error(f"Failed to load batch {batch_num}")

            success_rate = successful_batches / total_batches
            logger.info(
                f"Batch loading completed: {successful_batches}/{total_batches} "
                f"batches successful ({success_rate:.2%})"
            )
            return successful_batches == total_batches
        except Exception as e:
            logger.error(f"Error in batch loading: {e}")
            return False

    def copy_data_from_stage(
        self, table_name: str, stage_name: str, file_pattern: str = "*.parquet"
    ) -> bool:
        """
        Copy data from Snowflake stage using COPY command.

        Args:
            table_name: Target table name
            stage_name: Stage name
            file_pattern: File pattern to match

        Returns:
            bool: True if copy successful, False otherwise
        """
        try:
            copy_options = self.get_copy_options()
            copy_sql = (
                f"COPY INTO {table_name} FROM @{stage_name}/{file_pattern} "
                f"FILE_FORMAT = ({copy_options.get('FILE_FORMAT', 'PARQUET')}) "
                f"ON_ERROR = {copy_options.get('ON_ERROR', 'CONTINUE')} "
                f"PURGE = {copy_options.get('PURGE', True)} "
                f"RETURN_FAILED_ONLY = {copy_options.get('RETURN_FAILED_ONLY', True)}"
            )
            self.cursor.execute(copy_sql)
            logger.info(f"Data copied from stage {stage_name} to table {table_name}")
            return True
        except Exception as e:
            logger.error(f"Error copying data from stage: {e}")
            return False

    # Query and Performance Methods
    def execute_query_with_retry(
        self, query: str, params: Optional[tuple] = None, max_retries: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute query with retry logic.

        Args:
            query: SQL query string
            params: Optional query parameters
            max_retries: Optional max retries (uses config default if None)

        Returns:
            List[Dict[str, Any]]: Query results
        """
        max_retries = (
            self.config.max_retries
            if hasattr(self.config, "max_retries")
            else self.config.get("max_retries", 3)
        ) if max_retries is None else max_retries

        for attempt in range(max_retries + 1):
            try:
                return self.execute_query(query, params)
            except Exception as e:
                if attempt == max_retries:
                    logger.error(f"Query failed after {max_retries} retries: {e}")
                    raise
                logger.warning(f"Query attempt {attempt + 1} failed, retrying: {e}")
                time.sleep(self.config.get("retry_delay", 5))
        return []

    def get_query_performance_info(self, query_id: str) -> Dict[str, Any]:
        """
        Get query performance information.

        Args:
            query_id: Query ID

        Returns:
            Dict[str, Any]: Performance information
        """
        try:
            query = (
                "SELECT query_id, query_text, start_time, end_time, "
                "total_elapsed_time, bytes_scanned, rows_produced, warehouse_name "
                f"FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY WHERE query_id = '{query_id}'"
            )
            results = self.execute_query(query)
            return results[0 if results else {}]
        except Exception as e:
            logger.error(f"Error getting query performance info: {e}")
            return {}

    # Monitoring and Statistics Methods
    def get_warehouse_usage(self) -> Dict[str, Any]:
        """
        Get warehouse usage statistics.

        Returns:
            Dict[str, Any]: Warehouse usage information
        """
        try:
            # Try the full query first
            query = (
                "SELECT warehouse_name, start_time, end_time, credits_used, "
                "bytes_scanned, rows_produced "
                "FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_EVENTS_HISTORY "
                "WHERE start_time >= DATEADD(day, -7, CURRENT_TIMESTAMP()) "
                "ORDER BY start_time DESC LIMIT 100"
            )
            results = self.execute_query(query)
            return {
                "warehouse_events": results,
                "total_events": len(results),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.warning(f"Full warehouse usage query failed, trying simplified query: {e}")
            try:
                # Fallback to simplified query
                query = (
                    "SELECT warehouse_name, credits_used "
                    "FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_EVENTS_HISTORY "
                    "WHERE start_time >= DATEADD(day, -1, CURRENT_TIMESTAMP()) "
                    "LIMIT 10"
                )
                results = self.execute_query(query)
                return {
                    "warehouse_events": results,
                    "total_events": len(results),
                    "timestamp": datetime.now().isoformat(),
                    "note": "Simplified query due to permissions"
                }
            except Exception as fallback_e:
                logger.warning(f"Warehouse usage monitoring not available: {fallback_e}")
                return {
                    "warehouse_events": [],
                    "total_events": 0,
                    "timestamp": datetime.now().isoformat(),
                    "note": "Usage monitoring not available - may require higher privileges"
                }

    def get_database_usage(self) -> Dict[str, Any]:
        """
        Get database usage statistics.

        Returns:
            Dict[str, Any]: Database usage information
        """
        try:
            # Try the full query first
            query = (
                "SELECT database_name, schema_name, table_name, bytes, row_count "
                "FROM SNOWFLAKE.ACCOUNT_USAGE.TABLE_STORAGE_METRICS "
                "WHERE deleted = FALSE ORDER BY bytes DESC LIMIT 50"
            )
            results = self.execute_query(query)
            return {
                "storage_metrics": results,
                "total_tables": len(results),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.warning(f"Full database usage query failed, trying simplified query: {e}")
            try:
                # Fallback to simplified query
                query = (
                    "SELECT table_name, bytes "
                    "FROM SNOWFLAKE.ACCOUNT_USAGE.TABLE_STORAGE_METRICS "
                    "WHERE deleted = FALSE LIMIT 10"
                )
                results = self.execute_query(query)
                return {
                    "storage_metrics": results,
                    "total_tables": len(results),
                    "timestamp": datetime.now().isoformat(),
                    "note": "Simplified query due to permissions"
                }
            except Exception as fallback_e:
                logger.warning(f"Database usage monitoring not available: {fallback_e}")
                return {
                    "storage_metrics": [],
                    "total_tables": 0,
                    "timestamp": datetime.now().isoformat(),
                    "note": "Usage monitoring not available - may require higher privileges"
                }

    def get_warehouse_info(self) -> Dict[str, Any]:
        """
        Get current warehouse information.

        Returns:
            Dict[str, Any]: Warehouse information
        """
        try:
            query = "SELECT CURRENT_WAREHOUSE() as warehouse_name"
            results = self.execute_query(query)
            if results:
                warehouse_name = results[0].get("warehouse_name") or results[0].get(
                    "WAREHOUSE_NAME"
                )
                query = f"SHOW WAREHOUSES LIKE '{warehouse_name}'"
                warehouse_details = self.execute_query(query)
                return {
                    "warehouse_name": warehouse_name,
                    "details": warehouse_details,
                    "timestamp": datetime.now().isoformat(),
                }
            return {}
        except Exception as e:
            logger.error(f"Error getting warehouse info: {e}")
            return {}

    def get_database_info(self) -> Dict[str, Any]:
        """
        Get current database information.

        Returns:
            Dict[str, Any]: Database information
        """
        try:
            query = "SELECT CURRENT_DATABASE() as database_name"
            results = self.execute_query(query)
            if results:
                database_name = results[0].get("database_name") or results[0].get(
                    "DATABASE_NAME"
                )
                query = f"SHOW DATABASES LIKE '{database_name}'"
                database_details = self.execute_query(query)
                return {
                    "database_name": database_name,
                    "details": database_details,
                    "timestamp": datetime.now().isoformat(),
                }
            return {}
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            return {}

    def get_schema_info(self) -> Dict[str, Any]:
        """
        Get current schema information.

        Returns:
            Dict[str, Any]: Schema information
        """
        try:
            query = "SELECT CURRENT_SCHEMA() as schema_name"
            results = self.execute_query(query)
            if results:
                schema_name = results[0].get("schema_name") or results[0].get("SCHEMA_NAME")
                query = f"SHOW SCHEMAS LIKE '{schema_name}'"
                schema_details = self.execute_query(query)
                return {
                    "schema_name": schema_name,
                    "details": schema_details,
                    "timestamp": datetime.now().isoformat(),
                }
            return {}
        except Exception as e:
            logger.error(f"Error getting schema info: {e}")
            return {}

    def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get connection statistics.

        Returns:
            Dict[str, Any]: Connection statistics
        """
        return {
            "connection_active": self.connection is not None,
            "cursor_active": self.cursor is not None,
            "config_type": type(self.config).__name__,
            "warehouse_configured": self.warehouse_config is not None,
            "database_configured": self.database_config is not None,
            "timestamp": datetime.now().isoformat(),
        }

    def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Get ingestion statistics.

        Returns:
            Dict[str, Any]: Ingestion statistics
        """
        config = self.config if isinstance(self.config, dict) else vars(self.config)
        return {
            "account": config.get("account"),
            "database": config.get("database"),
            "schema": config.get("schema_name", config.get("schema")),
            "warehouse": config.get("warehouse"),
            "connection_initialized": self.connection is not None,
            "ingestion_timestamp": datetime.now().isoformat(),
        }
    
    def get_connection_params(self) -> Dict[str, Any]:
        """
        Get connection parameters from configuration.

        Returns:
            Dict[str, Any]: Connection parameters
        """
        if hasattr(self.config, "get_connection_params"):
            return self.config.get_connection_params()
        return {
            "account": self.config.get("account", "your-account.snowflakecomputing.com"),
            "user": self.config.get("user", "your-username"),
            "password": self.config.get("password", "your-password"),
            "warehouse": self.config.get("warehouse", "PBF_WAREHOUSE"),
            "database": self.config.get("database", "PBF_ANALYTICS"),
            "schema": self.config.get("schema_name", "RAW"),
            "role": self.config.get("role", "ACCOUNTADMIN"),
        }

    def get_session_params(self) -> Dict[str, Any]:
        """
        Get session parameters from configuration.

        Returns:
            Dict[str, Any]: Session parameters
        """
        if hasattr(self.config, "get_session_params"):
            return self.config.get_session_params()
        return {
            "QUERY_TAG": self.config.get("query_tag", "pbf-dev"),
            "WAREHOUSE_SIZE": self.config.get("warehouse_size", "SMALL"),
        }

    def get_copy_options(self) -> Dict[str, Any]:
        """
        Get COPY options from configuration.

        Returns:
            Dict[str, Any]: COPY options
        """
        if hasattr(self.config, "get_copy_options"):
            return self.config.get_copy_options()
        return {
            "FILE_FORMAT": "PARQUET",
            "ON_ERROR": "CONTINUE",
            "PURGE": True,
            "RETURN_FAILED_ONLY": True,
        }

    def load_data(
        self,
        data: List[Dict[str, Any]],
        table_name: str,
        source_name: str = "unknown",
        schema: str = "RAW",
        mode: str = "append",
        schema_factory=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Load data from operational databases into Snowflake data warehouse.
        
        This method provides ETL pipeline integration for loading data from
        operational databases (PostgreSQL, MongoDB, Cassandra, Redis, etc.)
        into Snowflake data warehouse, following the ETL architecture.
        
        IMPORTANT: This method requires schema_factory to handle predefined schemas
        and complex data sanitization that was proven to work in load_to_snowflake.py
        
        Args:
            data: List of records from operational databases
            table_name: Source table name (e.g., 'pbf_process_data', 'process_images')
            source_name: Source database name (postgresql, mongodb, etc.)
            schema: Target schema name (default: RAW)
            mode: Write mode (append, overwrite, ignore, error)
            schema_factory: SnowflakeSchemaFactory instance (REQUIRED)
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Loading results and statistics
        """
        try:
            logger.info(f"Loading operational data into Snowflake: {source_name}.{table_name}")
            
            # Initialize result tracking
            result = {
                "success": False,
                "records_loaded": 0,
                "records_processed": 0,
                "errors": [],
                "warnings": [],
                "table_name": table_name,
                "source_name": source_name,
                "schema": schema,
                "mode": mode
            }
            
            if not data:
                result["warnings"].append("No data to load")
                return result
            
            if not schema_factory:
                result["errors"].append("Schema factory is required for Snowflake loading")
                logger.error("Schema factory is required for Snowflake loading")
                return result
            
            result["records_processed"] = len(data)
            logger.info(f"Processing {len(data)} records from {source_name}")
            
            # Get schema mapping and predefined schema (CRITICAL for Snowflake)
            schema_mapping = self._get_schema_mapping()
            schema_key = schema_mapping.get(source_name, {}).get(table_name)
            
            if not schema_key:
                result["errors"].append(f"No schema mapping found for {source_name}.{table_name}")
                logger.error(f"No schema mapping found for {source_name}.{table_name}")
                return result
            
            # Get predefined schema and table name
            predefined_schema = schema_factory.get_schema(source_name, schema_key)
            snowflake_table = schema_factory.get_table_name(source_name, schema_key)
            
            if not predefined_schema:
                result["errors"].append(f"Schema not found in registry for {source_name}.{schema_key}")
                logger.error(f"Schema not found in registry for {source_name}.{schema_key}")
                return result
            
            logger.info(f"Using predefined schema for {snowflake_table}")
            
            # Create table with predefined schema (CRITICAL)
            try:
                create_sql = schema_factory.get_create_table_sql(source_name, schema_key, schema)
                self.execute_query(create_sql)
                logger.info(f"Created/verified table: {schema}.{snowflake_table}")
            except Exception as e:
                result["errors"].append(f"Failed to create table: {str(e)}")
                logger.error(f"Failed to create table {schema}.{snowflake_table}: {e}")
                return result
            
            # Prepare records for loading (add metadata)
            enriched_records = self._prepare_records_for_loading(data, source_name, table_name)
            
            # Get schema columns for comprehensive sanitization (CRITICAL)
            schema_columns = list(predefined_schema.keys())
            
            # Sanitize data values for Snowflake compatibility (CRITICAL)
            sanitized_records = []
            for record in enriched_records:
                sanitized_record = self._sanitize_record_for_snowflake(record, schema_columns)
                sanitized_records.append(sanitized_record)
            
            logger.info(f"Sanitized {len(sanitized_records)} records for Snowflake compatibility")
            
            # Load data using the proven load_data_batch method
            if self.load_data_batch(snowflake_table, sanitized_records):
                result["records_loaded"] = len(sanitized_records)
                result["success"] = True
                logger.info(f"Successfully loaded {len(sanitized_records)} records into {schema}.{snowflake_table}")
            else:
                result["errors"].append("Failed to load data batch")
                logger.error(f"Failed to load data into {schema}.{snowflake_table}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in load_data for table {schema}.{table_name}: {str(e)}")
            return {
                "success": False,
                "records_loaded": 0,
                "records_processed": 0,
                "errors": [str(e)],
                "warnings": [],
                "table_name": table_name,
                "source_name": source_name,
                "schema": schema,
                "mode": mode
            }
    
    def _prepare_records_for_loading(self, records: List[Dict[str, Any]], source_name: str, table_name: str) -> List[Dict[str, Any]]:
        """
        Prepare records for loading with metadata and proper data types.
        Based on the working prepare_records_for_loading function from load_to_snowflake.py
        """
        enriched_records = []
        
        for record in records:
            # Handle Neo4j Node objects that don't support item assignment
            if hasattr(record, 'items') and not isinstance(record, dict):
                # Convert Neo4j Node to dict first
                record_dict = dict(record)
            else:
                record_dict = record.copy() if isinstance(record, dict) else record
            
            # Add metadata
            record_dict['_source'] = source_name
            record_dict['_table'] = table_name
            record_dict['_loaded_at'] = datetime.now().isoformat()
            enriched_records.append(record_dict)
        
        return enriched_records
    
    def _get_schema_mapping(self):
        """
        Get mapping of table names to schema registry keys.
        Based on the working get_schema_mapping function from load_to_snowflake.py
        """
        return {
            # PostgreSQL schemas
            'postgresql': {
                'pbf_process_data': 'pbf_process_data',
                'powder_bed_data': 'powder_bed_data', 
                'ct_scan_data': 'ct_scan_data',
                'ispm_monitoring_data': 'ispm_monitoring_data',
                'powder_bed_defects': 'powder_bed_defects',
                'ct_scan_defect_types': 'ct_scan_defect_types',
                'pbf_process_statistics': 'pbf_process_statistics',
                'powder_bed_statistics': 'powder_bed_statistics',
                'ct_scan_dimensional_measurements': 'ct_scan_dimensional_measurements',
                'ct_scan_statistics': 'ct_scan_statistics',
                'ispm_sensor_statistics': 'ispm_sensor_statistics'
            },
            # MongoDB schemas
            'mongodb': {
                'process_images': 'mongodb_process_images',
                'ct_scan_images': 'mongodb_ct_scan_images',
                'powder_bed_images': 'mongodb_powder_bed_images',
                'machine_build_files': 'mongodb_machine_build_files',
                'model_3d_files': 'mongodb_model_3d_files',
                'raw_sensor_data': 'mongodb_raw_sensor_data',
                'process_logs': 'mongodb_process_logs',
                'machine_configurations': 'mongodb_machine_configurations'
            },
            # Cassandra schemas
            'cassandra': {
                'sensor_readings': 'sensor_readings',
                'machine_status': 'machine_status',
                'process_monitoring': 'process_monitoring',
                'alert_events': 'alert_events'
            },
            # Redis schemas
            'redis': {
                'process_cache': 'process_cache',
                'machine_status_cache': 'machine_status_cache',
                'sensor_readings_cache': 'sensor_readings_cache',
                'analytics_cache': 'analytics_cache',
                'job_queue_items': 'job_queue_items',
                'user_sessions': 'user_sessions'
            },
            # Elasticsearch schemas
            'elasticsearch': {
                'pbf_process': 'pbf_process',
                'sensor_readings': 'sensor_readings',
                'quality_metrics': 'quality_metrics',
                'machine_status': 'machine_status',
                'build_instructions': 'build_instructions',
                'analytics': 'analytics',
                'search_logs': 'search_logs'
            },
            # Neo4j schemas
            'neo4j': {
                'nodes': 'nodes',
                'relationships': 'relationships',
                'process_nodes': 'process_nodes',
                'machine_nodes': 'machine_nodes',
                'part_nodes': 'part_nodes',
                'sensor_nodes': 'sensor_nodes',
                'quality_nodes': 'quality_nodes',
                'material_nodes': 'material_nodes',
                'image_nodes': 'image_nodes',
                'file_nodes': 'build_file_nodes',
                'cache_nodes': 'process_cache_nodes',
                'queue_nodes': 'job_queue_nodes',
                'session_nodes': 'user_session_nodes',
                'reading_nodes': 'sensor_reading_nodes',
                'event_nodes': 'process_monitoring_nodes'
            }
        }
    
    def _sanitize_column_name(self, column_name):
        """
        Sanitize column names for Snowflake compatibility.
        Based on the working sanitize_column_name function from load_to_snowflake.py
        """
        # Replace problematic characters
        sanitized = column_name.replace('.', '_').replace(' ', '_').replace('-', '_').upper()
        
        # Handle specific field mappings first
        if sanitized == 'CURRENT':
            sanitized = 'CURRENT_AMPERAGE'
        
        # Handle reserved keywords and problematic identifiers
        reserved_keywords = [
            'FROM', 'TO', 'CURRENT', 'LOG_ID', 'ORDER', 'GROUP', 'SELECT', 'WHERE',
            'START_TIME', 'END_TIME', 'METADATA_SENSITIVITY', 'MODEL_ID', 'METADATA_DRIFT_RATE',
            'NODE_ID', 'RELATIONSHIP_ID', 'CACHE_KEY', 'USER_ID', 'BUILD_ID', 'PROCESS_ID',
            'FILE_FORMAT', 'LOG_TIMESTAMP', 'CONFIG_PARAMETERS', 'DASHBOARD_LAYOUT',
            'CREATED_AT', 'METADATA', 'QUALITY', 'IMAGE_WIDTH', 'FILE_PATH', 'TAGS',
            'PERFORMANCE_METRICS', 'ACTIVE_ALERTS', 'EVENT_DATA', 'PERCENTILES',
            'QUEUE_DATA', 'PERMISSIONS', 'DEFAULT_FILTERS', 'NOTIFICATION_SETTINGS',
            'PROCESS_PARAMETERS', 'MATERIAL_INFO', 'QUALITY_METRICS', 'SENSOR_LOCATION',
            'MEASUREMENT_DATA', 'ENVIRONMENTAL_CONDITIONS', 'QUALITY_FLAGS', 'PROCESS_CONTEXT',
            'DIMENSIONAL_METRICS', 'SURFACE_QUALITY', 'MECHANICAL_PROPERTIES', 'DEFECT_ANALYSIS',
            'QUALITY_SCORES', 'MEASUREMENT_CONDITIONS', 'MATERIAL_PROPERTIES', 'OPERATIONAL_STATE',
            'SYSTEM_HEALTH', 'LASER_SYSTEM', 'BUILD_PLATFORM', 'POWDER_SYSTEM', 'ALERTS_AND_WARNINGS',
            'MAINTENANCE_INFO', 'LAYER_INSTRUCTIONS', 'MATERIAL_REQUIREMENTS', 'QUALITY_REQUIREMENTS',
            'SUPPORT_STRUCTURES', 'POST_PROCESSING', 'SAFETY_REQUIREMENTS', 'PERFORMANCE_METRICS',
            'QUALITY_ANALYTICS', 'COST_ANALYTICS', 'TREND_ANALYSIS', 'PREDICTIVE_ANALYTICS',
            'ANOMALY_DETECTION', 'COMPARATIVE_ANALYSIS', 'KPI_METRICS', 'REPORTING_METADATA',
            'SEARCH_CONTEXT', 'SEARCH_RESULTS', 'USER_INTERACTION', 'ERROR_HANDLING',
            'SEARCH_ANALYTICS', 'TECHNICAL_DETAILS', 'LABELS', 'PROPERTIES', 'MAX_BUILD_VOLUME',
            'LAYER_THICKNESS_RANGE', 'DIMENSIONS', 'METRICS', 'STANDARDS', 'RANGE',
            'RESOLUTION', 'CAMERA_POSITION', 'TEMPERATURE_RANGE', 'FIELD_OF_VIEW',
            'HOT_SPOTS_DETECTED', 'COLD_SPOTS_DETECTED', 'VOXEL_SIZE', 'SCAN_RESOLUTION',
            'DEFECTS_DETECTED', 'MEASUREMENTS', 'TRAINING_COMPLETED', 'CERTIFICATIONS',
            'CONTACT_INFO', 'LOCATION', 'PARAMETERS', 'CURRENT', 'CURRENT_TIMESTAMP'
        ]
        if sanitized in reserved_keywords:
            sanitized = f"SF_{sanitized}"
        
        # Remove non-alphanumeric characters except underscores
        sanitized = ''.join(c if c.isalnum() or c == '_' else '_' for c in sanitized)
        
        # Limit length to 128 characters (Snowflake limit)
        if len(sanitized) > 128:
            sanitized = sanitized[:128]
        
        # Ensure it doesn't start with a number
        if sanitized and sanitized[0].isdigit():
            sanitized = f"COL_{sanitized}"
        
        return sanitized
    
    def _sanitize_data_value(self, value, column_name):
        """
        Sanitize data values for Snowflake compatibility.
        Based on the working sanitize_data_value function from load_to_snowflake.py
        """
        if value is None:
            return None
        
        # Handle complex objects first
        if isinstance(value, (dict, list)):
            import json
            try:
                return json.dumps(value)
            except:
                return str(value)
        
        # Handle boolean values - convert to proper format
        elif isinstance(value, bool):
            # Convert boolean to integer for NUMBER columns, string for others
            numeric_boolean_columns = [
                'MAINTENANCE_REQUIRED', 'ACKNOWLEDGED', 'DEFECTS_DETECTED', 
                'ARTIFACTS_DETECTED', 'CALIBRATION_STATUS', 'RESOLVED', 'ANOMALY_DETECTED'
            ]
            if column_name in numeric_boolean_columns:
                return 1 if value else 0  # Convert to integer for NUMBER columns
            else:
                return str(value).lower()  # Convert to string for other columns
        
        # Handle other problematic types
        elif hasattr(value, '__class__') and value.__class__.__name__ in ['ObjectId', 'Node']:
            return str(value)
        
        # Handle string values that should be VARIANT
        elif isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
            # Keep JSON strings as-is for VARIANT columns
            return value
        
        return value
    
    def _flatten_nested_data(self, data, prefix=''):
        """
        Flatten nested data structures to handle dotted notation issues.
        Based on the working flatten_nested_data function from load_to_snowflake.py
        """
        if not isinstance(data, dict):
            return {prefix: data} if prefix else data
        
        flattened = {}
        for key, value in data.items():
            new_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                flattened.update(self._flatten_nested_data(value, new_key))
            elif isinstance(value, list):
                # Handle lists by converting to JSON string
                import json
                try:
                    flattened[new_key] = json.dumps(value)
                except:
                    flattened[new_key] = str(value)
            else:
                flattened[new_key] = value
        
        return flattened
    
    def _sanitize_record_for_snowflake(self, record, schema_columns):
        """
        Comprehensive record sanitization for Snowflake compatibility.
        Based on the working sanitize_record_for_snowflake function from load_to_snowflake.py
        """
        sanitized_record = {}
        
        # First, flatten any nested structures
        flattened_record = self._flatten_nested_data(record)
        
        for key, value in flattened_record.items():
            # Sanitize column name
            sanitized_key = self._sanitize_column_name(key)
            
            # Handle missing fields by providing defaults
            if sanitized_key not in schema_columns:
                # Skip fields not in schema
                continue
                
            # Sanitize the value
            sanitized_value = self._sanitize_data_value(value, sanitized_key)
            
            # Handle specific problematic fields
            if sanitized_key == 'GRIDFS_FILE_ID' and sanitized_value is None:
                sanitized_value = 'N/A'  # Provide default for missing GridFS ID
            
            sanitized_record[sanitized_key] = sanitized_value
        
        # Ensure all required schema columns are present
        for column in schema_columns:
            if column not in sanitized_record:
                sanitized_record[column] = None
        
        return sanitized_record

    # Connection Management
    def close_connection(self) -> None:
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        logger.info("Snowflake connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_connection()