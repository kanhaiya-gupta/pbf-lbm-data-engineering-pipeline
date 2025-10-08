"""
Snowflake Client for PBF-LB/M Data Pipeline

This module provides Snowflake operations for data warehouse storage.
"""

import logging
import os
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from src.data_pipeline.config.storage_config import get_snowflake_config

logger = logging.getLogger(__name__)


class SnowflakeClient:
    """
    Snowflake client for data warehouse operations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Snowflake client.
        
        Args:
            config: Optional Snowflake configuration dictionary
        """
        self.config = config or get_snowflake_config()
        self.connection = None
        self.cursor = None
        self._initialize_connection()
    
    def _initialize_connection(self) -> None:
        """Initialize Snowflake connection."""
        try:
            import snowflake.connector
            
            self.connection = snowflake.connector.connect(
                account=self.config['account'],
                user=self.config['user'],
                password=self.config['password'],
                role=self.config.get('role', 'SYSADMIN'),
                warehouse=self.config.get('warehouse', 'COMPUTE_WH'),
                database=self.config.get('database', 'PBF_LBM_DW'),
                schema=self.config.get('schema', 'PUBLIC')
            )
            
            self.cursor = self.connection.cursor()
            logger.info("Snowflake connection initialized successfully")
            
        except ImportError:
            logger.error("snowflake-connector-python not available for Snowflake operations")
        except Exception as e:
            logger.error(f"Failed to initialize Snowflake connection: {e}")
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
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
        
        results = []
        
        try:
            # Execute query
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            
            # Get column names
            columns = [desc[0] for desc in self.cursor.description]
            
            # Fetch all results
            rows = self.cursor.fetchall()
            
            # Convert to list of dictionaries
            for row in rows:
                result = dict(zip(columns, row))
                results.append(result)
            
            logger.info(f"Executed query, returned {len(results)} rows")
            
        except Exception as e:
            logger.error(f"Error executing Snowflake query: {e}")
        
        return results
    
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
            # Get column names from first row
            columns = list(data[0].keys())
            columns_str = ", ".join(columns)
            
            # Build INSERT statement
            placeholders = ", ".join(["%s"] * len(columns))
            query = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"
            
            # Prepare data for insertion
            values_list = []
            for row in data:
                values = [row[col] for col in columns]
                values_list.append(values)
            
            # Execute batch insert
            self.cursor.executemany(query, values_list)
            
            logger.info(f"Inserted {len(data)} rows into Snowflake table: {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting data into Snowflake table {table_name}: {e}")
            return False
    
    def update_data(self, table_name: str, updates: Dict[str, Any], condition: str) -> bool:
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
            # Build UPDATE statement
            set_clause = ", ".join([f"{col} = %s" for col in updates.keys()])
            query = f"UPDATE {table_name} SET {set_clause} WHERE {condition}"
            
            # Execute update
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
            # Build DELETE statement
            query = f"DELETE FROM {table_name} WHERE {condition}"
            
            # Execute deletion
            self.cursor.execute(query)
            
            logger.info(f"Deleted data from Snowflake table: {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting data from Snowflake table {table_name}: {e}")
            return False
    
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
            
            schema_info = []
            for row in results:
                schema_info.append({
                    "column_name": row["name"],
                    "data_type": row["type"],
                    "null": row["null?"],
                    "default": row["default"],
                    "primary_key": row.get("primary key", False),
                    "unique_key": row.get("unique key", False)
                })
            
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
            query = f"SELECT COUNT(*) as count FROM {table_name}"
            results = self.execute_query(query)
            
            if results:
                count = results[0]["count"]
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
        info = {
            "table_name": table_name,
            "row_count": self.get_table_count(table_name),
            "schema": self.get_table_schema(table_name),
            "timestamp": datetime.now().isoformat()
        }
        
        return info
    
    def create_warehouse(self, warehouse_name: str, size: str = "SMALL", auto_suspend: int = 60, auto_resume: bool = True) -> bool:
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
            auto_resume_clause = "AUTO_RESUME = TRUE" if auto_resume else "AUTO_RESUME = FALSE"
            query = f"""
                CREATE WAREHOUSE {warehouse_name}
                WITH WAREHOUSE_SIZE = {size}
                AUTO_SUSPEND = {auto_suspend}
                {auto_resume_clause}
            """
            
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
    
    def create_schema(self, schema_name: str, database_name: Optional[str] = None, if_not_exists: bool = True) -> bool:
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
                warehouse_name = results[0]["warehouse_name"]
                
                # Get warehouse details
                query = f"SHOW WAREHOUSES LIKE '{warehouse_name}'"
                warehouse_details = self.execute_query(query)
                
                return {
                    "warehouse_name": warehouse_name,
                    "details": warehouse_details,
                    "timestamp": datetime.now().isoformat()
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
                database_name = results[0]["database_name"]
                
                # Get database details
                query = f"SHOW DATABASES LIKE '{database_name}'"
                database_details = self.execute_query(query)
                
                return {
                    "database_name": database_name,
                    "details": database_details,
                    "timestamp": datetime.now().isoformat()
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
                schema_name = results[0]["schema_name"]
                
                # Get schema details
                query = f"SHOW SCHEMAS LIKE '{schema_name}'"
                schema_details = self.execute_query(query)
                
                return {
                    "schema_name": schema_name,
                    "details": schema_details,
                    "timestamp": datetime.now().isoformat()
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting schema info: {e}")
            return {}
    
    def close_connection(self) -> None:
        """Close Snowflake connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        logger.info("Snowflake connection closed")
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Get ingestion statistics.
        
        Returns:
            Dict[str, Any]: Ingestion statistics
        """
        return {
            "account": self.config.get('account'),
            "database": self.config.get('database'),
            "schema": self.config.get('schema'),
            "warehouse": self.config.get('warehouse'),
            "connection_initialized": self.connection is not None,
            "ingestion_timestamp": datetime.now().isoformat()
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_connection()
