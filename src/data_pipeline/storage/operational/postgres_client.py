"""
PostgreSQL Client for PBF-LB/M Data Pipeline

This module provides PostgreSQL operations for operational storage.
"""

import logging
import os
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from src.data_pipeline.config.postgres_config import get_postgres_config

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
            
            # Handle both dictionary and Pydantic config
            if isinstance(self.config, dict):
                host = self.config.get('host', 'localhost')
                port = self.config.get('port', 5432)
                database = self.config.get('database', 'lpbf_research')
                user = self.config.get('user', 'postgres')
                password = self.config.get('password', 'password')
            else:
                # Pydantic model
                host = self.config.host
                port = self.config.port
                database = self.config.database
                user = self.config.username
                password = self.config.password
            
            self.connection = psycopg2.connect(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password
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
            
            # Check if this is a DDL statement (CREATE, DROP, ALTER, etc.)
            query_upper = query.strip().upper()
            is_ddl = any(keyword in query_upper for keyword in ['CREATE', 'DROP', 'ALTER', 'INSERT', 'UPDATE', 'DELETE'])
            
            if is_ddl:
                # DDL statements don't return results - this is expected
                logger.debug(f"Executed DDL statement successfully")
            else:
                # Only fetch results for SELECT queries
                try:
                    rows = self.cursor.fetchall()
                    # RealDictCursor already returns dictionaries
                    results.extend(rows)
                    logger.debug(f"Executed query, returned {len(results)} rows")
                except Exception as fetch_error:
                    # Some queries might not have results to fetch
                    logger.debug(f"No results to fetch for query: {fetch_error}")
            
        except Exception as e:
            logger.error(f"Error executing PostgreSQL query: {e}")
            # Rollback the transaction to allow subsequent queries
            try:
                self.connection.rollback()
            except Exception as rollback_error:
                logger.error(f"Error during rollback: {rollback_error}")
        
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
    
    def commit(self) -> bool:
        """
        Commit the current transaction.
        
        Returns:
            bool: True if commit successful, False otherwise
        """
        if not self.connection:
            logger.error("PostgreSQL connection not initialized")
            return False
        
        try:
            self.connection.commit()
            logger.info("Transaction committed successfully")
            return True
        except Exception as e:
            logger.error(f"Error committing transaction: {e}")
            return False
    
    def rollback(self) -> bool:
        """
        Rollback the current transaction.
        
        Returns:
            bool: True if rollback successful, False otherwise
        """
        if not self.connection:
            logger.error("PostgreSQL connection not initialized")
            return False
        
        try:
            self.connection.rollback()
            logger.info("Transaction rolled back successfully")
            return True
        except Exception as e:
            logger.error(f"Error rolling back transaction: {e}")
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
            
            # CRITICAL: Commit the transaction to persist the data
            self.connection.commit()
            
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
    
    def load_data(
        self,
        df: Any,
        table_name: str,
        mode: str = "append",
        batch_size: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Load Spark DataFrame data into PostgreSQL table.
        
        This method provides ETL pipeline integration for loading transformed
        Spark DataFrames into PostgreSQL, following the ETL architecture.
        
        Args:
            df: Spark DataFrame from transform modules
            table_name: Target table name
            mode: Write mode (append, overwrite, ignore, error)
            batch_size: Batch size for processing
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Loading results and statistics
        """
        try:
            logger.info(f"Loading Spark DataFrame into PostgreSQL table: {table_name}")
            
            # Initialize result tracking
            result = {
                "success": False,
                "records_loaded": 0,
                "records_processed": 0,
                "errors": [],
                "warnings": [],
                "table_name": table_name,
                "mode": mode,
                "batch_size": batch_size
            }
            
            # Convert Spark DataFrame to list of dictionaries
            data_list = self._convert_spark_dataframe(df)
            if not data_list:
                result["warnings"].append("No data to load")
                return result
            
            result["records_processed"] = len(data_list)
            logger.info(f"Converted {len(data_list)} records from Spark DataFrame")
            
            # Handle different modes
            if mode == "overwrite":
                # Drop and recreate table (if table exists)
                self.drop_table(table_name, if_exists=True)
                logger.info(f"Dropped existing table: {table_name}")
            
            # Batch processing
            total_loaded = 0
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i:i + batch_size]
                
                try:
                    # Insert batch using existing method
                    success = self.insert_data(table_name, batch)
                    if success:
                        total_loaded += len(batch)
                        logger.info(f"Loaded batch {i//batch_size + 1}: {len(batch)} records")
                    else:
                        result["errors"].append(f"Failed to load batch {i//batch_size + 1}")
                        
                except Exception as e:
                    error_msg = f"Error loading batch {i//batch_size + 1}: {str(e)}"
                    result["errors"].append(error_msg)
                    logger.error(error_msg)
            
            # Update result
            result["records_loaded"] = total_loaded
            result["success"] = total_loaded > 0 and len(result["errors"]) == 0
            
            if result["success"]:
                logger.info(f"Successfully loaded {total_loaded} records into {table_name}")
            else:
                logger.error(f"Failed to load data into {table_name}. Errors: {result['errors']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in load_data for table {table_name}: {str(e)}")
            return {
                "success": False,
                "records_loaded": 0,
                "records_processed": 0,
                "errors": [str(e)],
                "warnings": [],
                "table_name": table_name,
                "mode": mode,
                "batch_size": batch_size
            }
    
    def _convert_spark_dataframe(self, df: Any) -> List[Dict[str, Any]]:
        """
        Convert Spark DataFrame to list of dictionaries for PostgreSQL insertion.
        
        Args:
            df: Spark DataFrame from transform modules
            
        Returns:
            List[Dict[str, Any]]: Converted data list
        """
        try:
            if hasattr(df, 'collect'):
                # Spark DataFrame - convert to list of dicts
                rows = df.collect()
                data_list = []
                
                for row in rows:
                    # Convert Row to dictionary
                    row_dict = row.asDict()
                    
                    # Handle Spark-specific data types
                    processed_dict = self._process_spark_row(row_dict)
                    data_list.append(processed_dict)
                
                return data_list
                
            elif isinstance(df, list):
                # Already a list of dictionaries
                return df
                
            elif isinstance(df, dict):
                # Single dictionary
                return [df]
                
            else:
                logger.warning(f"Unsupported DataFrame type: {type(df)}")
                return []
                
        except Exception as e:
            logger.error(f"Error converting Spark DataFrame: {str(e)}")
            return []
    
    def _process_spark_row(self, row_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process Spark Row data to PostgreSQL-compatible format.
        
        Args:
            row_dict: Dictionary from Spark Row
            
        Returns:
            Dict[str, Any]: Processed dictionary
        """
        try:
            processed = {}
            
            for key, value in row_dict.items():
                # Handle None values
                if value is None:
                    processed[key] = None
                # Handle Spark-specific types
                elif hasattr(value, 'isoformat'):
                    # Datetime objects
                    processed[key] = value
                elif isinstance(value, (int, float, str, bool)):
                    # Basic types
                    processed[key] = value
                elif hasattr(value, '__dict__'):
                    # Complex objects - convert to JSON string
                    import json
                    processed[key] = json.dumps(value.__dict__, default=str)
                else:
                    # Fallback - convert to string
                    processed[key] = str(value)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing Spark row: {str(e)}")
            return row_dict  # Return original if processing fails

    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_connection()
