"""
PostgreSQL Client for PBF-LB/M Data Pipeline

This module provides PostgreSQL operations for operational storage.
"""

import logging
import os
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from src.data_pipeline.config.storage_config import get_postgres_config

logger = logging.getLogger(__name__)


class PostgresClient:
    """
    PostgreSQL client for operational database operations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize PostgreSQL client.
        
        Args:
            config: Optional PostgreSQL configuration dictionary
        """
        self.config = config or get_postgres_config()
        self.connection = None
        self.cursor = None
        self._initialize_connection()
    
    def _initialize_connection(self) -> None:
        """Initialize PostgreSQL connection."""
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            
            self.connection = psycopg2.connect(
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password']
            )
            
            self.cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            logger.info("PostgreSQL connection initialized successfully")
            
        except ImportError:
            logger.error("psycopg2 not available for PostgreSQL operations")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL connection: {e}")
    
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
            logger.error("PostgreSQL cursor not initialized")
            return []
        
        results = []
        
        try:
            # Execute query
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            
            # Fetch all results
            rows = self.cursor.fetchall()
            
            # Convert to list of dictionaries
            for row in rows:
                results.append(dict(row))
            
            logger.info(f"Executed query, returned {len(results)} rows")
            
        except Exception as e:
            logger.error(f"Error executing PostgreSQL query: {e}")
        
        return results
    
    def execute_transaction(self, queries: List[str], params_list: Optional[List[tuple]] = None) -> bool:
        """
        Execute multiple queries in a transaction.
        
        Args:
            queries: List of SQL query strings
            params_list: Optional list of query parameters
            
        Returns:
            bool: True if transaction successful, False otherwise
        """
        if not self.cursor:
            logger.error("PostgreSQL cursor not initialized")
            return False
        
        try:
            # Begin transaction
            self.cursor.execute("BEGIN")
            
            # Execute queries
            params_list = params_list or [None] * len(queries)
            for query, params in zip(queries, params_list):
                if params:
                    self.cursor.execute(query, params)
                else:
                    self.cursor.execute(query)
            
            # Commit transaction
            self.cursor.execute("COMMIT")
            logger.info(f"Executed transaction with {len(queries)} queries")
            return True
            
        except Exception as e:
            # Rollback transaction
            try:
                self.cursor.execute("ROLLBACK")
            except:
                pass
            
            logger.error(f"Error executing PostgreSQL transaction: {e}")
            return False
    
    def create_table(self, table_name: str, schema: str, if_not_exists: bool = True) -> bool:
        """
        Create a table in PostgreSQL.
        
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
            logger.info(f"Created PostgreSQL table: {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating PostgreSQL table {table_name}: {e}")
            return False
    
    def drop_table(self, table_name: str, if_exists: bool = True) -> bool:
        """
        Drop a table in PostgreSQL.
        
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
            logger.info(f"Dropped PostgreSQL table: {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error dropping PostgreSQL table {table_name}: {e}")
            return False
    
    def insert_data(self, table_name: str, data: List[Dict[str, Any]]) -> bool:
        """
        Insert data into a PostgreSQL table.
        
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
            
            logger.info(f"Inserted {len(data)} rows into PostgreSQL table: {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting data into PostgreSQL table {table_name}: {e}")
            return False
    
    def update_data(self, table_name: str, updates: Dict[str, Any], condition: str) -> bool:
        """
        Update data in a PostgreSQL table.
        
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
            
            logger.info(f"Updated data in PostgreSQL table: {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating data in PostgreSQL table {table_name}: {e}")
            return False
    
    def delete_data(self, table_name: str, condition: str) -> bool:
        """
        Delete data from a PostgreSQL table.
        
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
            
            logger.info(f"Deleted data from PostgreSQL table: {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting data from PostgreSQL table {table_name}: {e}")
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
            query = """
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    character_maximum_length,
                    numeric_precision,
                    numeric_scale
                FROM information_schema.columns 
                WHERE table_name = %s
                ORDER BY ordinal_position
            """
            
            results = self.execute_query(query, (table_name,))
            
            schema_info = []
            for row in results:
                schema_info.append({
                    "column_name": row["column_name"],
                    "data_type": row["data_type"],
                    "is_nullable": row["is_nullable"] == "YES",
                    "column_default": row["column_default"],
                    "character_maximum_length": row["character_maximum_length"],
                    "numeric_precision": row["numeric_precision"],
                    "numeric_scale": row["numeric_scale"]
                })
            
            logger.info(f"Retrieved schema for PostgreSQL table: {table_name}")
            return schema_info
            
        except Exception as e:
            logger.error(f"Error getting schema for PostgreSQL table {table_name}: {e}")
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
                logger.info(f"PostgreSQL table {table_name} has {count} rows")
                return count
            
            return 0
            
        except Exception as e:
            logger.error(f"Error getting count for PostgreSQL table {table_name}: {e}")
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
    
    def create_index(self, table_name: str, index_name: str, columns: List[str], unique: bool = False) -> bool:
        """
        Create an index on a table.
        
        Args:
            table_name: Name of the table
            index_name: Name of the index
            columns: List of column names for the index
            unique: Whether to create a unique index
            
        Returns:
            bool: True if index creation successful, False otherwise
        """
        try:
            unique_clause = "UNIQUE" if unique else ""
            columns_str = ", ".join(columns)
            query = f"CREATE {unique_clause} INDEX {index_name} ON {table_name} ({columns_str})"
            
            self.cursor.execute(query)
            logger.info(f"Created index {index_name} on PostgreSQL table {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating index {index_name} on PostgreSQL table {table_name}: {e}")
            return False
    
    def drop_index(self, index_name: str) -> bool:
        """
        Drop an index.
        
        Args:
            index_name: Name of the index
            
        Returns:
            bool: True if index drop successful, False otherwise
        """
        try:
            query = f"DROP INDEX {index_name}"
            
            self.cursor.execute(query)
            logger.info(f"Dropped index: {index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error dropping index {index_name}: {e}")
            return False
    
    def create_sequence(self, sequence_name: str, start_value: int = 1, increment: int = 1, max_value: Optional[int] = None, min_value: Optional[int] = None) -> bool:
        """
        Create a sequence.
        
        Args:
            sequence_name: Name of the sequence
            start_value: Starting value
            increment: Increment value
            max_value: Maximum value
            min_value: Minimum value
            
        Returns:
            bool: True if sequence creation successful, False otherwise
        """
        try:
            query = f"CREATE SEQUENCE {sequence_name} START WITH {start_value} INCREMENT BY {increment}"
            
            if max_value:
                query += f" MAXVALUE {max_value}"
            if min_value:
                query += f" MINVALUE {min_value}"
            
            self.cursor.execute(query)
            logger.info(f"Created sequence: {sequence_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating sequence {sequence_name}: {e}")
            return False
    
    def drop_sequence(self, sequence_name: str, if_exists: bool = True) -> bool:
        """
        Drop a sequence.
        
        Args:
            sequence_name: Name of the sequence
            if_exists: Whether to use IF EXISTS clause
            
        Returns:
            bool: True if sequence drop successful, False otherwise
        """
        try:
            if_exists_clause = "IF EXISTS" if if_exists else ""
            query = f"DROP SEQUENCE {if_exists_clause} {sequence_name}"
            
            self.cursor.execute(query)
            logger.info(f"Dropped sequence: {sequence_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error dropping sequence {sequence_name}: {e}")
            return False
    
    def get_table_list(self, schema: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of tables.
        
        Args:
            schema: Optional schema name to filter by
            
        Returns:
            List[Dict[str, Any]]: List of table information
        """
        try:
            if schema:
                query = """
                    SELECT 
                        table_name,
                        table_schema,
                        table_type
                    FROM information_schema.tables 
                    WHERE table_schema = %s
                    ORDER BY table_name
                """
                results = self.execute_query(query, (schema,))
            else:
                query = """
                    SELECT 
                        table_name,
                        table_schema,
                        table_type
                    FROM information_schema.tables 
                    WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
                    ORDER BY table_schema, table_name
                """
                results = self.execute_query(query)
            
            tables = []
            for row in results:
                tables.append({
                    "name": row["table_name"],
                    "schema": row["table_schema"],
                    "type": row["table_type"]
                })
            
            logger.info(f"Retrieved {len(tables)} PostgreSQL tables")
            return tables
            
        except Exception as e:
            logger.error(f"Error getting PostgreSQL table list: {e}")
            return []
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get database information.
        
        Returns:
            Dict[str, Any]: Database information
        """
        try:
            # Get database name
            query = "SELECT current_database() as database_name"
            db_result = self.execute_query(query)
            
            # Get database size
            query = "SELECT pg_size_pretty(pg_database_size(current_database())) as database_size"
            size_result = self.execute_query(query)
            
            # Get connection info
            query = "SELECT current_user as current_user, inet_server_addr() as server_address, inet_server_port() as server_port"
            conn_result = self.execute_query(query)
            
            info = {
                "database_name": db_result[0]["database_name"] if db_result else "unknown",
                "database_size": size_result[0]["database_size"] if size_result else "unknown",
                "current_user": conn_result[0]["current_user"] if conn_result else "unknown",
                "server_address": conn_result[0]["server_address"] if conn_result else "unknown",
                "server_port": conn_result[0]["server_port"] if conn_result else "unknown",
                "timestamp": datetime.now().isoformat()
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting PostgreSQL database info: {e}")
            return {}
    
    def close_connection(self) -> None:
        """Close PostgreSQL connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        logger.info("PostgreSQL connection closed")
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Get ingestion statistics.
        
        Returns:
            Dict[str, Any]: Ingestion statistics
        """
        return {
            "host": self.config.get('host'),
            "port": self.config.get('port'),
            "database": self.config.get('database'),
            "user": self.config.get('user'),
            "connection_initialized": self.connection is not None,
            "ingestion_timestamp": datetime.now().isoformat()
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_connection()
